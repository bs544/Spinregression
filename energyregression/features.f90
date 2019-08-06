! l should be from 0!

module features
    use config
    use io, only : error_message
    use boundaries, only : find_neighbouring_images
    use utility, only : load_balance_alg_1,sort_array,python_int

    implicit none

    !* blas/lapack
    real(8),external :: ddot

    contains
        subroutine calculate_local(cell,atom_positions,rcut,weightings,parallel,&
        &nmax,calc_type,buffer_size,X)
            ! Compute bispectrum features as in [1]
            !
            ! x_nl = sum_{m=-l}^{m=l} c_{nlm}^* c_{nlm}
            !
            ! Arguments
            ! ---------
            ! cell             : shape=(3,3),     units=cartesians
            ! atom_positions   : shape=(3,Natm),  units=fractional coordinates
            ! weighting        : shape=(Natm),    no units
            !
            ! Note
            ! ----
            ! This routine uses full periodic image convention, not nearest
            ! image
            !
            ! [1] PHYSICAL REVIEW B 87, 184115 (2013)
            use omp_lib

            implicit none

            !* args
            real(8),intent(in) :: cell(1:3,1:3),atom_positions(:,:)
            real(8),intent(in) :: rcut
            logical,intent(in) :: parallel
            real(8),intent(in) :: weightings(:)
            integer,intent(in) :: nmax,calc_type,buffer_size
            real(8),intent(inout) :: X(:,:)


            !* scratch
            integer :: dim(1:2),natm,ii,loop(1:2)
            real(8),allocatable :: neigh_images(:,:),dists(:),neigh_weights(:)

            !* openmp
            integer :: thread_idx,num_threads

            !===============!
            !* arg parsing *!
            !===============!

            dim=shape(atom_positions)

            !* number of atoms in unit cell
            natm = dim(2)

            !* check shape of output array (Nfeats,ngrid)
            dim = shape(X)
            if ((dim(1).ne.nmax.or.(dim(2).ne.natm))) then
                call error_message("calculate_bispectrum_type1","shape mismatch between output array and input args")
            end if

            !* calculation type
            call feature_param_type__set_calc_type(feature_param,calc_type)

            !* interaction cut off
            call feature_param_type__set_rcut(feature_param,rcut)

            !* n(l) max
            call feature_param_type__set_n(feature_param,nmax)

            !* initialise shared config info
            call config_type__set_cell(cell)
            call config_type__set_local_positions(atom_positions)
            call config_type__set_atom_weights(weightings)

            !* cartesians of all relevant atom images

            !* list of relevant images
            call find_neighbouring_images(neigh_images,buffer_size)

            !* generate cartesians of all relevant atoms
            call config_type__generate_ultracell(neigh_images)

            !* remove redunancy
            call init_buffer_all_general()

            if (.not.parallel) then
                loop(1) = 1
                loop(2) = natm

                X = 0.0d0

                do ii=loop(1),loop(2),1
                    !* generate [[r,theta,phi] for atom neighbouring frid point ii]
                    call config_type__generate_neighbouring_distances(ii,dists,neigh_weights)

                    if(.not.allocated(dists)) then
                        !* no atoms within local approximation
                        X(:,ii) = 0.0d0
                    else

                        !* get type1 features
                        call features_bispectrum_type1(dists,X(:,ii),neigh_weights)
                    end if
                end do
            else
                !$omp parallel num_threads(omp_get_max_threads()),&
                !$omp& default(shared),&
                !$omp& private(thread_idx,dists,ii,loop,num_threads,neigh_weights),&
                !$omp& copyin(buffer_cnlm)

                !* [0,num_threads-1]
                thread_idx = omp_get_thread_num()

                num_threads = omp_get_max_threads()

                !* evenly split workload
                call load_balance_alg_1(thread_idx,num_threads,natm,loop)

                do ii=loop(1),loop(2),1
                    !* generate [[r,theta,phi] for atom neighbouring frid point ii]
                    call config_type__generate_neighbouring_distances(ii,dists,neigh_weights)

                    if(.not.allocated(dists)) then
                        !* no atoms within local approximation
                        X(:,ii) = 0.0d0
                    else
                        call features_bispectrum_type1(dists,X(:,ii),neigh_weights)
                    end if
                end do

                !$omp end parallel
            end if
        end subroutine calculate_local

        subroutine features_bispectrum_type1(dists,x,neigh_weights)
            implicit none

            !* args ! CHANGE INOUT TO IN
            real(8),intent(inout) :: dists(:)
            real(8),intent(inout) :: x(:)
            real(8),intent(inout) :: neigh_weights(:)

            !* scratch
            integer :: cntr,dim(1:2)
            integer :: nn
            real(8) :: res_real
            complex(8) :: buffer(1:2)

            !* redundancy arrays specific to grid point
            call init_buffer_all_polar(dists,neigh_weights)


            cntr = 1
            res_real = 0.0d0
            X = 0.0d0
            if (feature_param%calc_type.eq.0) then
                do nn=1,feature_param%nmax

                    res_real = buffer_cnlm(nn)

                    X(cntr) = res_real
                    cntr = cntr + 1

                end do
            end if


        end subroutine features_bispectrum_type1

        subroutine init_buffer_all_general()
            implicit none


            call init_buffer_radial_phi_Nalpha()
            call init_buffer_radial_basis_overlap()
            call init_buffer_cnlm_mem()


        end subroutine init_buffer_all_general

        subroutine init_buffer_cnlm_mem()
            implicit none


            if(allocated(buffer_cnlm)) then
                deallocate(buffer_cnlm)
            end if

            allocate(buffer_cnlm(1:feature_param%nmax))
        end subroutine init_buffer_cnlm_mem

        subroutine init_buffer_radial_phi_Nalpha()
            implicit none

            !* scratch
            integer :: nn

            if (allocated(buffer_radial_phi_Nalpha)) then
                deallocate(buffer_radial_phi_Nalpha)
            end if
            allocate(buffer_radial_phi_Nalpha(feature_param%nmax))

            do nn=1,feature_param%nmax,1
                buffer_radial_phi_Nalpha(nn) = 1.0d0/sqrt(feature_param%rcut**(2*nn+5) / dble(2*nn+5))
            end do
        end subroutine init_buffer_radial_phi_Nalpha

        subroutine init_buffer_all_polar(dists,neigh_weights)
            ! initialise and compute all redundancy arrays for info specific
            ! to a particular density grid point
            implicit none

            real(8),intent(in) :: dists(:)
            real(8),intent(in) :: neigh_weights(:)

            !* radial component in radial bases
            call init_radial_g(dists)

            !* compute cnlm
            call calc_cnlm(neigh_weights)
        end subroutine init_buffer_all_polar

        real(8) function radial_phi_type1(alpha,dr)
            implicit none

            integer,intent(in) :: alpha
            real(8),intent(in) :: dr

            radial_phi_type1 = (feature_param%rcut - dr)**(alpha+2) * buffer_radial_phi_Nalpha(alpha)
        end function radial_phi_type1

        subroutine init_radial_g(dists)
            implicit none

            !* args
            real(8),intent(in) :: dists(:)

            !* scratch
            integer :: dim(1),Nneigh,ii,nn
            real(8),allocatable :: phi(:,:)

            dim = shape(dists)
            Nneigh = dim(1)

            allocate(phi(Nneigh,feature_param%nmax))
            if (allocated(buffer_radial_g)) then
                deallocate(buffer_radial_g)
            end if
            allocate(buffer_radial_g(Nneigh,feature_param%nmax))

            do nn=1,feature_param%nmax,1
                do ii=1,Nneigh,1
                    phi(ii,nn) = radial_phi_type1(nn,dists(ii))
                end do
            end do

            !* g_i,beta = sum_alpha W_beta,alpha phi_i,alpha
            call dgemm('n','n',Nneigh,feature_param%nmax,feature_param%nmax,1.0d0,phi,Nneigh,&
            &buffer_radial_overlap_w,feature_param%nmax,0.0d0,buffer_radial_g,Nneigh)
        end subroutine init_radial_g

        subroutine init_buffer_radial_basis_overlap()
            ! compute radial basis overlap matrix S_ij = <phi_i | phi_j>
            ! and the square root inverse, W = S^{-1/2}
            use utility, only : sqrt_of_matrix_inverse

            implicit none

            integer :: ii,jj

            if(allocated(buffer_radial_overlap_s)) then
                deallocate(buffer_radial_overlap_s)
            end if
            if(allocated(buffer_radial_overlap_w)) then
                deallocate(buffer_radial_overlap_w)
            end if
            allocate(buffer_radial_overlap_s(feature_param%nmax,feature_param%nmax))
            allocate(buffer_radial_overlap_w(feature_param%nmax,feature_param%nmax))

            !* overlap matrix S_ij = <phi_i | phi_j>
            do ii=1,feature_param%nmax,1
                do jj=1,feature_param%nmax,1
                    buffer_radial_overlap_s(ii,jj) = sqrt( (5.0d0+2.0d0*dble(ii))*(5.0d0+2.0d0*dble(jj)) ) /&
                    & (5.0d0 + dble(ii) + dble(jj))
                end do
            end do

            !* cmopute W = S^{-1/2}
            call sqrt_of_matrix_inverse(buffer_radial_overlap_s,buffer_radial_overlap_w)
        end subroutine init_buffer_radial_basis_overlap

        real(8) function ddot_wrapper(arr1,arr2)
            implicit none

            real(8),intent(in) :: arr1(:),arr2(:)

            integer :: dim1(1:1),dim2(1:1)
            real(8) :: res2

            dim1 = shape(arr1)
            dim2 = shape(arr2)

            if (dim1(1).ne.dim2(1)) then
                call error_message("ddot_wrapper","shapes not consistent")
            end if

            ! CANNOT SLICE arrays for ddot with blas, need to pass slice
            ! through arg list
            res2 = ddot(dim1(1),arr1,1,arr2,1)
            ddot_wrapper = res2
        end function ddot_wrapper

        subroutine calc_cnlm(neigh_weights)
            ! compute cnlm for given density point
            implicit none
            real(8),intent(in) :: neigh_weights(:)
            integer :: nn,dim(1:2),ii,natm
            complex(8) :: res,tmp


            dim = shape(buffer_radial_g)
            natm = dim(1)

            buffer_cnlm = 0.0d0
            do nn=1,feature_param%nmax
                res = 0.0d0
                tmp = 0.0d0
                do ii=1,natm
                    tmp = neigh_weights(ii)*buffer_radial_g(ii,nn)
                    res = res + tmp
                end do

                buffer_cnlm(nn) = res
            end do

        end subroutine calc_cnlm

end module features
