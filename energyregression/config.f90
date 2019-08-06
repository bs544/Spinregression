module config
    use io, only : error_message

    implicit none

    real(8),external :: dnrm2
    external :: dgemm

    !* type definitions

    type config_type
        real(8) :: cell(1:3,1:3)                            ! real space (A)
        real(8),allocatable :: local_positions(:,:)         ! fractional coordinates /(A)
        real(8),allocatable :: all_positions(:,:)           ! cartesian coordinates of relevant atom images (A)
        real(8),allocatable :: weightings(:)                ! weightings of atoms
        real(8),allocatable :: all_weightings(:)            ! weightings of all relevant atomm images
        integer :: nall                                     ! atoms in ultracell
    end type config_type


    type feature_param_type
        !---------------------------------------------------!
        ! NOTE                                              !
        ! ----                                              !
        ! attribute calc_type determines which features     !
        ! from powerspectrum and bispectrum to compute.     !
        !                                                   !
        ! calc_type = 0 : powerspectrum only                !
        ! calc_type = 1 : bispectrum only                   !
        ! calc_type = 2 : powerspectrum and bispectrum      !
        !---------------------------------------------------!

        real(8) :: rcut                                     ! interaction cut off (A)
        integer :: nmax                                     ! radial component
        integer :: calc_type                                ! type of feature calc to perform
    end type feature_param_type

    !* type instances

    type(config_type),public :: structure                   ! instance of configuration info
    type(feature_param_type),public :: feature_param        ! instance of bispectrum parameters

    !* buffer arrays to remove redunant computation


    real(8),allocatable :: buffer_radial_phi_Nalpha(:)      ! normalizing constant for alpha
    real(8),allocatable :: buffer_radial_g(:,:)             ! radial components for given grid point
    real(8),allocatable :: buffer_radial_overlap_s(:,:)     ! overlap matrix of radial bases
    real(8),allocatable :: buffer_radial_overlap_w(:,:)     ! linear combination coefficients from basis overlap
    complex(8),allocatable :: buffer_cnlm(:)                ! environment projection onto bases

    !* omp directives for private globally scoped variables
    !$omp threadprivate(buffer_radial_g)
    !$omp threadprivate(buffer_cnlm)


    contains
        !* methods

        subroutine feature_param_type__set_rcut(type_instance,rcut)
            implicit none

            type(feature_param_type),intent(inout) :: type_instance
            real(8),intent(in) :: rcut

            type_instance%rcut = rcut
        end subroutine feature_param_type__set_rcut

        subroutine feature_param_type__set_calc_type(type_instance,calc_type)
            implicit none

            type(feature_param_type),intent(inout) :: type_instance
            integer,intent(in) :: calc_type

            if ((calc_type.lt.0).or.(calc_type.gt.2)) then
                call error_message("feature_param_type__set_calc_type","unsupported calculation type")
            end if
            type_instance%calc_type = calc_type
        end subroutine feature_param_type__set_calc_type

        subroutine feature_param_type__set_n(type_instance,nmax)
            implicit none

            type(feature_param_type),intent(inout) :: type_instance
            integer,intent(in) :: nmax


            if (nmax.le.0) then
                call error_message("feature_param_type__set_n","must give nmax>0")
            end if


            type_instance%nmax = nmax
        end subroutine feature_param_type__set_n

        subroutine config_type__generate_ultracell(neigh_images)
            implicit none

            !* args
            real(8),intent(in) :: neigh_images(:,:)

            !* scratch
            integer :: dim(1:2),ncells,natm,ii,jj,nall
            real(8),allocatable :: arraycopy(:,:)

            dim = shape(neigh_images)
            ncells = dim(2)

            if(.not.allocated(structure%local_positions)) then
                call error_message("config_type__generate_ultracell","initialise local_positions")
            end if
            dim = shape(structure%local_positions)
            natm = dim(2)

            nall = natm*ncells

            if(allocated(structure%all_positions)) deallocate(structure%all_positions)
            allocate(structure%all_positions(1:3,1:nall))

            if(allocated(structure%all_weightings)) deallocate(structure%all_weightings)
            allocate(structure%all_weightings(1:nall))

            do ii=1,ncells,1
                do jj=1,natm,1
                    !* fractional coordinates
                    structure%all_positions(:,(ii-1)*natm+jj) = structure%local_positions(:,jj) +&
                    &neigh_images(:,ii)
                    structure%all_weightings((ii-1)*natm+jj) = structure%weightings(jj)
                end do
            end do

            !* can't use read+write to same var in lapack/blas
            allocate(arraycopy(1:3,1:nall))
            arraycopy = structure%all_positions

            !* cartesians
            call dgemm('n','n',3,nall,3,1.0d0,structure%cell,3,arraycopy,3,0.0d0,&
            &structure%all_positions,3)

            structure%nall = nall

            deallocate(arraycopy)
        end subroutine config_type__generate_ultracell

        subroutine config_type__set_cell(cell)
            implicit none

            real(8),intent(in) :: cell(1:3,1:3)

            structure%cell = cell
        end subroutine config_type__set_cell

        subroutine config_type__set_local_positions(positions)
            implicit none

            !* args
            real(8),intent(in) :: positions(:,:)

            !* scratch
            integer :: dim(1:2)

            if (allocated(structure%local_positions)) then
                deallocate(structure%local_positions)
            end if

            dim = shape(positions)

            allocate(structure%local_positions(dim(1),dim(2)))
            structure%local_positions = positions

            !* wrap positions back to local cell (0<= fractional coords <=1)
            call config_type__wrap_atom_positions()
        end subroutine config_type__set_local_positions

        subroutine config_type__set_atom_weights(weightings)
            implicit none

            !* args
            real(8),intent(in) :: weightings(:)

            !* scratch
            integer :: dim(1)

            if (allocated(structure%weightings)) then
                deallocate(structure%weightings)
            end if

            dim = shape(weightings)

            allocate(structure%weightings(dim(1)))
            structure%weightings = weightings
        
        end subroutine config_type__set_atom_weights

        subroutine config_type__generate_neighbouring_distances(idx,dists,neigh_weights)
            !* Y_lm(theta,phi) = k_lm P_m(cos(theta)) * exp(i phi)

            !use, intrinsic :: ieee_arithmetic

            implicit none

            !* args
            integer, intent(in) :: idx
            real(8),allocatable,intent(inout) :: dists(:)
            real(8),allocatable,intent(inout) :: neigh_weights(:)

            !* scratch
            real(8) :: gridpoint(1:3)
            real(8) :: rcut2,dr_vec(1:3), amin, rad2
            real(8) :: dist_buffer(1:structure%nall)
            real(8) :: weights_buffer(1:structure%nall)
            real(8), allocatable :: local_cartesians(:,:)
            integer :: ii,cntr, natm, dim(1:2)

            if(allocated(dists)) then
                deallocate(dists)
            end if

            weights_buffer = 0.0d0

            dim = shape(structure%local_positions)
            natm = dim(2)

            allocate(local_cartesians(1:3,1:natm))
            call dgemm('n','n',3,natm,3,1.0d0,structure%cell,3,structure%local_positions,3,0.0d0,&
            &local_cartesians,3)

            if(.not.allocated(structure%all_positions)) then
                call error_message("config_type__generate_neighbouring_polar","attribute all_positions not allocated")
            end if
            rcut2 = feature_param%rcut**2
            amin = dble(1e-8)**2

            cntr = 0
            gridpoint = local_cartesians(:,idx)
            do ii=1,structure%nall,1
                !* cartesian displacement
                
                dr_vec = structure%all_positions(:,ii) - gridpoint

                rad2 = sum(dr_vec**2)
                ! print *, rad2

                if (rad2.le.rcut2.and.(rad2.gt.amin)) then
                    cntr = cntr + 1
                    weights_buffer(cntr) = structure%all_weightings(ii)
                    dist_buffer(cntr) = sqrt(rad2)
                end if
            end do

            if (cntr.gt.0) then
                if(allocated(neigh_weights)) then
                    deallocate(neigh_weights)
                end if
                allocate(neigh_weights(1:cntr))
                allocate(dists(cntr))
                neigh_weights = 1.0d0
                dists(:) = dist_buffer(1:cntr)
                neigh_weights(1:cntr) = weights_buffer(1:cntr)
            end if
        end subroutine config_type__generate_neighbouring_distances

        subroutine config_type__wrap_atom_positions()
            ! wrap fractional coordinates of atoms to within [0,1]
            implicit none

            !* scratch
            integer :: dim(1:2),ii,jj

            structure%local_positions = structure%local_positions - floor(structure%local_positions)

            dim = shape(structure%local_positions)
            do ii=1,dim(2)
                do jj=1,3
                    if ((structure%local_positions(jj,ii).lt.0.0d0).or.&
                    &(structure%local_positions(jj,ii).gt.1.0d0)) then
                        !* sanity check
                        call error_message("config_type__wrap_atom_positions","failed to wrap positions")
                    end if
                end do
            end do
        end subroutine config_type__wrap_atom_positions

end module config
