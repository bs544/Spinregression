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
        integer :: nall                                     ! atoms in ultracell
    end type config_type


    type bispect_param_type
        real(8) :: rcut                                     ! interaction cut off (A)
        integer :: lmax                                     ! spherical component
        integer :: nmax                                     ! radial component
    end type bispect_param_type

    !* type instances

    type(config_type),public :: structure                   ! instance of configuration info
    type(bispect_param_type),public :: bispect_param        ! instance of bispectrum parameters

    !* buffer arrays to remove redunant computation
    
    real(8),allocatable :: buffer_spherical_harm_const(:,:) ! constant to spherical harmonics (m,l)
    real(8),allocatable :: buffer_radial_phi_Nalpha(:)      ! normalizing constant for alpha       
    real(8),allocatable :: buffer_radial_g(:,:)             ! radial components for given grid point
    real(8),allocatable :: buffer_spherical_p(:,:,:)        ! associated legendre polynomial
    real(8),allocatable :: buffer_polar_sc(:,:)             ! [cos(theta),sin(theta) for neighbour for grid point]
    real(8),allocatable :: buffer_radial_overlap_s(:,:)     ! overlap matrix of radial bases
    real(8),allocatable :: buffer_radial_overlap_w(:,:)     ! linear combination coefficients from basis overlap

    contains
        !* methods
        
        subroutine bispect_param_type__set_rcut(type_instance,rcut)
            implicit none

            type(bispect_param_type),intent(inout) :: type_instance
            real(8),intent(in) :: rcut
            
            type_instance%rcut = rcut
        end subroutine bispect_param_type__set_rcut

        subroutine bispect_param_type__set_ln(type_instance,lmax,nmax)
            implicit none

            type(bispect_param_type),intent(inout) :: type_instance
            integer,intent(in) :: lmax,nmax

            if (lmax.lt.0) then
                call error_message("bispect_param_type__set_ln","must give lmax>0")
            else if (nmax.le.0) then
                call error_message("bispect_param_type__set_ln","must give nmax>0")
            end if

            type_instance%lmax = lmax
            type_instance%nmax = nmax
        end subroutine bispect_param_type__set_ln

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

            do ii=1,ncells,1
                do jj=1,natm,1
                    !* fractional coordinates
                    structure%all_positions(:,(ii-1)*natm+jj) = structure%local_positions(:,jj) +&
                    &neigh_images(:,ii)
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
    
        subroutine config_type__generate_neighbouring_polar(gridpoint,polar)
            use, intrinsic :: ieee_arithmetic            

            implicit none

            !* args
            real(8),intent(in) :: gridpoint(1:3)
            real(8),allocatable,intent(inout) :: polar(:,:)

            !* scratch
            real(8) :: rcut2,dr_vec(1:3)
            real(8) :: polar_buffer(1:3,1:structure%nall)
            integer :: ii,cntr

            if(allocated(polar)) then
                deallocate(polar)
            end if
            if(.not.allocated(structure%all_positions)) then
                call error_message("config_type__generate_neighbouring_polar","attribute all_positions not allocated")
            end if
            rcut2 = bispect_param%rcut**2

            cntr = 0
            do ii=1,structure%nall,1
                !* cartesian displacement
                dr_vec = structure%all_positions(:,ii) - gridpoint
                
                if (sum(dr_vec**2).le.rcut2) then
                    cntr = cntr + 1
                    polar_buffer(1,cntr) = dnrm2(3,dr_vec,1)
                    polar_buffer(2,cntr) = dr_vec(3) / polar_buffer(1,cntr)
                    if (.not.ieee_is_finite(polar_buffer(2,cntr))) then 
                        !* r=0, use (phi,theta) = 0 in this case
                        polar_buffer(2,cntr) = 0.0d0
                        polar_buffer(3,cntr) = 0.0d0
                    else     
                        polar_buffer(2,cntr) = acos(polar_buffer(2,cntr))               
                        polar_buffer(3,cntr) = atan(dr_vec(2)/dr_vec(1))
                    end if
                end if 
            end do

            if (cntr.gt.0) then
                allocate(polar(3,cntr))
                polar(:,:) = polar_buffer(:,1:cntr)
            end if
        end subroutine config_type__generate_neighbouring_polar

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
