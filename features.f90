! l should be from 0!

module features
    use config
    use io, only : error_message
    use boundaries, only : find_neighbouring_images
    use utility, only : load_balance_alg_1

    implicit none
   
    !* blas/lapack 
    real(8),external :: ddot

    contains
        integer function check_cardinality(lmax,nmax,calc_type)
            implicit none

            integer,intent(in) :: lmax,nmax,calc_type

            integer :: res

            if (calc_type.eq.0) then
                !* powerspectrum only, l=[0,lmax] , n=[1,nmax]
                res = (lmax+1)*nmax
            else if (calc_type.eq.1) then
                !* bispectrum only, l1=[0,lmax],l2=[0,lmax],l3=[0,lmax]
                res = (lmax+1)**3 * nmax
            else if (calc_type.eq.2) then
                res = (lmax+1)*nmax + (lmax+1)**3 * nmax
            else
                call error_message("check_cardinality","unsupported calculation type")
            end if
            check_cardinality = res
        end function check_cardinality

        subroutine calculate_powerspectrum_type1(cell,atom_positions,grid_coordinates,rcut,parallel,&
        &lmax,nmax,calc_type,X)
            ! Compute bispectrum features as in [1]
            !
            ! x_nl = sum_{m=-l}^{m=l} c_{nlm}^* c_{nlm} 
            !
            ! Arguments
            ! ---------
            ! cell             : shape=(3,3),     units=cartesians
            ! atom_positions   : shape=(3,Natm),  units=fractional coordinates
            ! grid_coordinates : shape=(3,Ngrid), units=cartesians
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
            real(8),intent(in) :: grid_coordinates(:,:),rcut
            logical,intent(in) :: parallel
            integer,intent(in) :: lmax,nmax,calc_type
            real(8),intent(inout) :: X(:,:)

            !* scratch
            integer :: dim(1:2),natm,ngrid,ii,loop(1:2)
            real(8),allocatable :: neigh_images(:,:),polar(:,:)

            !* openmp
            integer :: thread_idx,num_threads

            !===============!
            !* arg parsing *!
            !===============!

            dim=shape(atom_positions)
        
            !* number of atoms in unit cell
            natm = dim(2)

            dim = shape(grid_coordinates)
        
            !* number of  density points to calculate features for
            ngrid = dim(2)

            !* check shape of output array (Nfeats,ngrid)
            dim = shape(X)
            if ((dim(1).ne.check_cardinality(lmax,nmax,calc_type)).or.(dim(2).ne.ngrid)) then
                call error_message("calculate_bispectrum_type1","shape mismatch between output array and input args")
            end if

            !* calculation type
            call bispect_param_type__set_calc_type(bispect_param,calc_type)

            !* interaction cut off
            call bispect_param_type__set_rcut(bispect_param,rcut)

            !* n(l) max
            call bispect_param_type__set_ln(bispect_param,lmax,nmax)

            !* initialise shared config info
            call config_type__set_cell(cell)
            call config_type__set_local_positions(atom_positions)   

            !* cartesians of all relevant atom images 
            
            !* list of relevant images
            call find_neighbouring_images(neigh_images)
            
            !* generate cartesians of all relevant atoms
            call config_type__generate_ultracell(neigh_images)
       
            !* remove redunancy 
            call init_buffer_all_general()

            if (.not.parallel) then
                loop(1) = 1
                loop(2) = ngrid

                do ii=loop(1),loop(2),1
                    !* generate [[r,theta,phi] for atom neighbouring frid point ii]
                    call config_type__generate_neighbouring_polar(grid_coordinates(:,ii),polar)

                    if(.not.allocated(polar)) then
                        !* no atoms within local approximation
                        X(:,ii) = 0.0d0
                    else
                        !* get type1 features
                        call features_bispectrum_type1(polar,X(:,ii))
                    end if
                end do
            else
                !$omp parallel num_threads(omp_get_max_threads()),&
                !$omp& default(shared),&
                !$omp& private(thread_idx,polar,ii,loop,num_threads)
            
                !* [0,num_threads-1]
                thread_idx = omp_get_thread_num()
                
                num_threads = omp_get_max_threads()

                !* evenly split workload
                call load_balance_alg_1(thread_idx,num_threads,ngrid,loop)
                
                do ii=loop(1),loop(2),1
                    !* generate [[r,theta,phi] for atom neighbouring frid point ii]
                    call config_type__generate_neighbouring_polar(grid_coordinates(:,ii),polar)

                    if(.not.allocated(polar)) then
                        !* do atoms within local approximation
                        X(:,ii) = 0.0d0
                    else
                        !* get type1 features
                        call features_bispectrum_type1(polar,X(:,ii)) 
                    end if
                end do
                
                !$omp end parallel
            end if            
        end subroutine calculate_powerspectrum_type1

        subroutine features_bispectrum_type1_deprecated(polar,x)
            ! concacenation order:
            !
            ! for l in [0,lmax]:
            !     for n in [1,nmax]:
            implicit none

            !* args
            real(8),intent(in) :: polar(:,:)
            real(8),intent(inout) :: x(:)
            
            !* scratch
            integer :: dim(1:2),cntr
            integer :: Nneigh,ll,nn,ii,mm
            real(8) :: val_ln,reduce_array(1:2)
            real(8) :: tmp1,tmp2(1:2),tmp3

            dim = shape(polar)

            !* number of neighbouring atoms to grid point
            Nneigh = dim(2)

            !* redundancy arrays specific to grid point
            call init_buffer_all_polar(polar)
            
            cntr = 1
            do ll=0,bispect_param%lmax,1
                do nn=1,bispect_param%nmax,1   
                    val_ln = 0.0d0
     
                    ! reduce page thrashing later
                    tmp3 = buffer_spherical_harm_const(0,ll)
            
                    do mm=1,ll 
                        reduce_array = 0.0d0

                        neighbour_loop : do ii=1,Nneigh,1
                            tmp1 = buffer_radial_g(ii,nn)*buffer_spherical_p(ii,mm,ll)

                            tmp2(1) = tmp1*buffer_polar_sc(1,ii,mm)    ! cos(m phi)
                            tmp2(2) = tmp1*buffer_polar_sc(2,ii,mm)    ! sin(m phi)

                            reduce_array = reduce_array + tmp2
                        end do neighbour_loop
                        
                        val_ln = val_ln + buffer_spherical_harm_const(mm,ll)*sum(reduce_array**2)
                    end do
                    
                    !* count -,+mm
                    val_ln = val_ln*2.0d0
                    
                    !* m=0 contribution : cos(m phi)=1,sin(m phi)=0
                    val_ln = val_ln + ddot_wrapper(buffer_radial_g(:,nn),buffer_spherical_p(:,0,ll))**2 * tmp3

                    x(cntr) = val_ln

! DEBUG FOR NEW IMPLEMENTATION
val_ln = 0.0d0 
do mm=1,ll
    val_ln = val_ln + abs(buffer_cnlm(mm,ll,nn)*dconjg(buffer_cnlm(mm,ll,nn)))
end do
val_ln = val_ln*2.0d0 + abs( buffer_cnlm(0,ll,nn) * dconjg(buffer_cnlm(0,ll,nn)) )
write(*,*) 'old = ',x(cntr),'new=',val_ln

! DEBUG FOR NEW IMPLEMENTATION
                    cntr = cntr + 1
                end do
            end do
        end subroutine features_bispectrum_type1_deprecated
        
        subroutine features_bispectrum_type1(polar,x)
            ! concacenation order:
            !
            ! for l in [0,lmax]:
            !     for n in [1,nmax]:
            use spherical_harmonics, only : cg_varshalovich,sph_harm
            implicit none

            !* args ! CHANGE INOUT TO IN
            real(8),intent(inout) :: polar(:,:)
            real(8),intent(inout) :: x(:)
            
            !* scratch
            integer :: cntr,lmax
            integer :: ll,nn
            integer :: ll_1,ll_2,ll_3,mm_1,mm_2,mm_3
            real(8) :: res_real,cg_coeff
            complex(8) :: buffer(1:2),res_cmplx
integer :: dim(1:2),natm,ii,mm
dim = shape(polar)
natm = dim(2)
!polar = 1.0d0            
            !* redundancy arrays specific to grid point
            call init_buffer_all_polar(polar)
           
            lmax = bispect_param%lmax

            cntr = 1
            if ((bispect_param%calc_type.eq.0).or.(bispect_param%calc_type.eq.2)) then
                do nn=1,bispect_param%nmax
                    do ll=0,lmax
                        res_real = abs( buffer_cnlm(0,ll,nn) * dconjg(buffer_cnlm(0,ll,nn)) )
                        res_real = res_real + sum(abs( buffer_cnlm(1:ll,ll,nn) * dconjg(buffer_cnlm(1:ll,ll,nn)) ))*2.0d0

                        X(cntr) = res_real
                        cntr = cntr + 1
                    end do
                end do
            else if ((bispect_param%calc_type.eq.1).or.(bispect_param%calc_type.eq.2)) then
            
                do nn=1,bispect_param%nmax
                    do ll=0,lmax
                        do mm=-ll,ll
                            res_cmplx = complex(0.0d0,0.0d0)
                            do ii=1,natm
                                res_cmplx = res_cmplx + 1.0d0*sph_harm(mm,ll,polar(2,ii),polar(3,ii))
                            end do
                            buffer_cnlm(mm,ll,nn) = res_cmplx
                        end do
                    end do
                end do
                
                nn_loop : do nn=1,bispect_param%nmax
                    ll_1_loop : do ll_1=0,lmax
                        ll_2_loop : do ll_2=0,lmax
                            ll_3_loop : do ll_3=0,lmax
                                
                                res_cmplx = complex(0.0d0,0.0d0)
                                mm_1_loop : do mm_1=-ll_1,ll_1
                                    buffer(1) = dconjg(buffer_cnlm(mm_1,ll_1,nn))

                                    mm_2_loop : do mm_2=-ll_2,ll_2
                                        buffer(2) = buffer(1)*buffer_cnlm(mm_2,ll_2,nn)

                                        mm_3_loop : do mm_3=-ll_3,ll_3
                                            !cg_coeff = buffer_cg_coeff(mm_3,mm_2,mm_1,ll_3,ll_2,ll_1)
                                            cg_coeff = cg_varshalovich(dble(ll_2),dble(mm_2),dble(ll_3),dble(mm_3),&
                                            &dble(ll_1),dble(mm_1))
                                            
                                            !res_cmplx = res_cmplx + buffer(2)*buffer_cnlm(mm_3,ll_3,nn) * cg_coeff

                                            res_cmplx = res_cmplx + cg_coeff*conjg(buffer_cnlm(mm_1,ll_1,nn))*&
                                            &buffer_cnlm(mm_2,ll_2,nn)*buffer_cnlm(mm_3,ll_3,nn)
                                        end do mm_3_loop
                                    end do mm_2_loop
                                end do mm_1_loop

!if(abs(imagpart(res_cmplx)).gt.1e-10) then
!write(*,*) res_cmplx,ll_3,ll_2,ll_1
!end if
                                X(cntr) = real(res_cmplx)
                                cntr = cntr + 1
                            end do ll_3_loop
                        end do ll_2_loop
                    end do ll_1_loop
                end do nn_loop
            end if
        end subroutine features_bispectrum_type1

        subroutine init_buffer_all_general()
            implicit none

            call init_buffer_spherical_harm_const()
            call init_buffer_radial_phi_Nalpha()
            call init_buffer_radial_basis_overlap()
            call init_buffer_cnlm_mem()

            if ((bispect_param%calc_type.eq.1).or.(bispect_param%calc_type.eq.2)) then
                !* CB coefficients repeatedly use factorial evaluation
                call init_buffer_factorial()

                !* bispectrum needs Clebsch-Gordan coefficients
                call init_buffer_cg_coeff()
            end if
        end subroutine init_buffer_all_general

        subroutine init_spherical_harm(polars)
            use spherical_harmonics, only : plgndr_s

            implicit none

            !* args
            real(8),intent(in) :: polars(:,:)

            !* scratch
            integer :: dim(1:2),ii,mm,ll
            real(8),allocatable :: cos_theta(:)
            real(8) :: cos_m,sin_m,const_ml

            dim = shape(polars)
            allocate(cos_theta(1:dim(2)))
            do ii=1,dim(2),1
                cos_theta(ii) = cos(polars(2,ii))
            end do

            if (allocated(buffer_spherical_harm)) then
                deallocate(buffer_spherical_harm)
            end if
            allocate(buffer_spherical_harm(1:dim(2),0:bispect_param%lmax,0:bispect_param%lmax))
            buffer_spherical_harm = 0.0d0

            do ll=0,bispect_param%lmax
                !* norm const for ((m,l)=(0,l)
                !const_ml = buffer_spherical_harm_const(0,ll)
                const_ml = sqrt(buffer_spherical_harm_const(0,ll)) !! THIS IS **2 the ACTUAL SPH HARM CONSTANT
                do ii=1,dim(2)
                    buffer_spherical_harm(ii,0,ll) = plgndr_s(ll,0,cos_theta(ii))*const_ml*complex(1.0d0,0.0d0)
                end do

                do mm=1,ll
                    !* normalisation term
                    !const_ml = buffer_spherical_harm_const(mm,ll)
                    const_ml = sqrt(buffer_spherical_harm_const(mm,ll))

                    do ii=1,dim(2),1
                        cos_m = buffer_polar_sc(1,ii,mm)
                        sin_m = buffer_polar_sc(2,ii,mm)
                        buffer_spherical_harm(ii,mm,ll) = plgndr_s(ll,mm,cos_theta(ii))*const_ml*complex(cos_m,sin_m)
                    end do
                end do
            end do

            deallocate(cos_theta)
        end subroutine init_spherical_harm

        subroutine init_buffer_cg_coeff()
            use spherical_harmonics, only : cg_su2,cg_varshalovich

            implicit none

            !* scratch
            integer :: lmax,ll_1,ll_2,ll_3,mm_1,mm_2,mm_3
            real(8) :: cgval,dble_ll_1,dble_mm_1,dble_ll_2,dble_mm_2,dble_ll_3,dble_mm_3

            if(allocated(buffer_cg_coeff)) then
                deallocate(buffer_cg_coeff)
            end if

            lmax = bispect_param%lmax

            ! indices = (m3,m2,m,l3,l2,l)
            allocate(buffer_cg_coeff(-lmax:lmax,-lmax:lmax,-lmax:lmax,0:lmax,0:lmax,0:lmax))

            buffer_CG_coeff = 0.0d0

            do ll_1=0,lmax
                dble_ll_1 = dble(ll_1)
                do ll_2=0,lmax
                    dble_ll_2 = dble(ll_2)
                    do ll_3=0,lmax
                        dble_ll_3 = dble(ll_3)
                        do mm_1=-ll_1,ll_1
                            dble_mm_1 = dble(mm_1)
                            do mm_2 = -ll_2,ll_2
                                dble_mm_2 = dble(mm_2)
                                do mm_3=-ll_3,ll_3
                                    dble_mm_3 = dble(mm_3)
                                    !* compute
                                    cgval = cg_varshalovich(dble_ll_2,dble_mm_2,dble_ll_3,dble_mm_3,dble_ll_1,dble_mm_1)
                                    !cgval = cg_su2(ll_1,ll_2,ll_3,mm_1,mm_2,mm_3)
                                    
                                    !* store
                                    buffer_cg_coeff(mm_3,mm_2,mm_1,ll_3,ll_2,ll_1) = cgval 
                                end do
                            end do
                        end do
                    end do
                end do
            end do

        end subroutine init_buffer_CG_coeff

        subroutine init_buffer_spherical_harm_const()
            implicit none

            !* scratch
            integer :: ll,mm

            if(allocated(buffer_spherical_harm_const)) then
                deallocate(buffer_spherical_harm_const)
            end if
            
            allocate(buffer_spherical_harm_const(0:bispect_param%lmax,0:bispect_param%lmax))
            buffer_spherical_harm_const = 0.0d0            

            do ll=0,bispect_param%lmax,1
                do mm=0,ll
                    buffer_spherical_harm_const(mm,ll) = spherical_harm_const__sub1(mm,ll)
                end do
            end do
        end subroutine init_buffer_spherical_harm_const

        subroutine init_buffer_cnlm_mem()
            implicit none

            !* scratch
            integer :: lmax

            lmax = bispect_param%lmax

            if(allocated(buffer_cnlm)) then
                deallocate(buffer_cnlm)
            end if

            allocate(buffer_cnlm(-lmax:lmax,0:lmax,1:bispect_param%nmax))
        end subroutine init_buffer_cnlm_mem

        subroutine init_buffer_radial_phi_Nalpha()
            implicit none

            !* scratch
            integer :: nn

            if (allocated(buffer_radial_phi_Nalpha)) then
                deallocate(buffer_radial_phi_Nalpha)
            end if
            allocate(buffer_radial_phi_Nalpha(bispect_param%nmax))

            do nn=1,bispect_param%nmax,1
                buffer_radial_phi_Nalpha(nn) = 1.0d0/sqrt(bispect_param%rcut**(2*nn+5) / dble(2*nn+5))
            end do
        end subroutine init_buffer_radial_phi_Nalpha


        subroutine init_buffer_spherical_p(polar)
            ! calculate associated legendre polynomail for cos(theta_j) for all
            ! atoms j
            use spherical_harmonics, only : plgndr_s

            implicit none

            !* args
            real(8),intent(in) :: polar(:,:)

            !* scratch
            integer :: dim(1:2)
            integer :: Nneigh,mm,ll,ii
            real(8),allocatable :: cos_vals(:)

            !* number of neighbouring atoms
            dim = shape(polar)
            Nneigh = dim(2)

            if(allocated(buffer_spherical_p)) then
                deallocate(buffer_spherical_p)
            end if
            allocate(buffer_spherical_p(1:Nneigh,0:bispect_param%lmax,0:bispect_param%lmax))

            !* cos(theta_i) for all i needs computing only once
            allocate(cos_vals(1:Nneigh))
            do ii=1,Nneigh
                cos_vals(ii) = cos(polar(2,ii))
            end do

            !*              m       l
            !* [1,Nneigh][0,lmax][0,lmax]
            buffer_spherical_p = 0.0d0

            do ll=0,bispect_param%lmax,1
                do mm=0,ll,1
                    do ii=1,Nneigh,1
                        buffer_spherical_p(ii,mm,ll) = plgndr_s(ll,mm,cos_vals(ii))
                    end do
                end do
            end do
        end subroutine init_buffer_spherical_p

        subroutine init_buffer_polar_sc(polar)
            implicit none

            !* args
            real(8),intent(in) :: polar(:,:)

            !* scratch
            integer :: dim(1:2),Nneigh,ii,mm
            real(8) :: dble_mm

            dim = shape(polar)

            Nneigh = dim(2)

            if(allocated(buffer_polar_sc)) then
                deallocate(buffer_polar_sc)
            end if
            if(bispect_param%lmax.ge.1) then
                allocate(buffer_polar_sc(1:2,1:Nneigh,1:bispect_param%lmax))
               
                do mm=1,bispect_param%lmax
                    dble_mm = dble(mm)
                    do ii=1,Nneigh,1
                        !* (cos(m phi),sin(m phi))
                        buffer_polar_sc(1,ii,mm) = cos(polar(3,ii)*dble_mm)
                        buffer_polar_sc(2,ii,mm) = sin(polar(3,ii)*dble_mm)
                    end do
                end do
            end if
        end subroutine init_buffer_polar_sc

        subroutine init_buffer_all_polar(polar)
            ! initialise and compute all redundancy arrays for info specific
            ! to a particular density grid point
            implicit none

            real(8),intent(in) :: polar(:,:)            

            !* associated legendre polynomials of polar angle - NO LONGER NECESSARY
            call init_buffer_spherical_p(polar)

            !* trigonometric bases for azimuthal angle
            call init_buffer_polar_sc(polar)
            
            !* Y_iml
            call init_spherical_harm(polar)
        
            !* radial component in radial bases
            call init_radial_g(polar)

            !* compute cnlm
            call calc_cnlm(polar)
        end subroutine init_buffer_all_polar

        real(8) function spherical_harm_const__sub1(mm,ll)
            implicit none
    
            !* args
            integer,intent(in) :: ll,mm

            !* scratch
            real(8) :: dble_ll,dble_mm

            dble_ll = dble(ll)
            dble_mm = dble(mm)
        
            spherical_harm_const__sub1 = ((2.0d0*dble_ll+1.0d0)*dble(factorial(ll-mm)))/(12.5663706144d0*dble(factorial(ll+mm)))
        end function spherical_harm_const__sub1

        integer recursive function factorial(x) result(res)
            implicit none

            !* args
            integer,intent(in) :: x
    
            if (x.eq.0) then
                res = 1
            else if (x.gt.0) then
                res = x*factorial(x-1)
            else
                call error_message("factorial","negative argument passed to factorial")
            end if
        end function factorial

        real(8) function radial_phi_type1(alpha,dr)
            implicit none

            integer,intent(in) :: alpha
            real(8),intent(in) :: dr

            radial_phi_type1 = (bispect_param%rcut - dr)**(alpha+2) * buffer_radial_phi_Nalpha(alpha)
        end function radial_phi_type1

        subroutine init_radial_g(polar)
            implicit none

            !* args
            real(8),intent(in) :: polar(:,:)
    
            !* scratch
            integer :: dim(1:2),Nneigh,ii,nn
            real(8),allocatable :: phi(:,:)

            dim = shape(polar)
            Nneigh = dim(2)

            allocate(phi(Nneigh,bispect_param%nmax))
            if (allocated(buffer_radial_g)) then
                deallocate(buffer_radial_g)
            end if
            allocate(buffer_radial_g(Nneigh,bispect_param%nmax))

            do nn=1,bispect_param%nmax,1
                do ii=1,Nneigh,1
                    phi(ii,nn) = radial_phi_type1(nn,polar(1,ii))
                end do
            end do        

            !* g_i,beta = sum_alpha W_beta,alpha phi_i,alpha
            call dgemm('n','n',Nneigh,bispect_param%nmax,bispect_param%nmax,1.0d0,phi,Nneigh,&
            &buffer_radial_overlap_w,bispect_param%nmax,0.0d0,buffer_radial_g,Nneigh)
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
            allocate(buffer_radial_overlap_s(bispect_param%nmax,bispect_param%nmax))
            allocate(buffer_radial_overlap_w(bispect_param%nmax,bispect_param%nmax))

            !* overlap matrix S_ij = <phi_i | phi_j>
            do ii=1,bispect_param%nmax,1
                do jj=1,bispect_param%nmax,1
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

        subroutine calc_cnlm(polar)
            ! compute cnlm for given density point
use spherical_harmonics, only : sph_harm            
            implicit none
            real(8) :: polar(:,:)
            integer :: lmax,mm,ll,nn,dim(1:3),ii,natm
            complex(8) :: res(1:2),tmp


            lmax = bispect_param%lmax
            dim = shape(buffer_spherical_harm)
            natm = dim(1)
            
            buffer_cnlm = 0.0d0
            do nn=1,bispect_param%nmax
                do ll=0,lmax
                    do mm=-ll,ll
                        res(1) = complex(0.0d0,0.0d0)
                        do ii=1,natm
                            res(1) = res(1) + 1.0d0*sph_harm(mm,ll,polar(2,ii),polar(3,ii))
                        end do
                        buffer_cnlm(mm,ll,nn) = res(1)
                    end do

                    !do mm=0,ll
                    !    res = 0.0d0
                    !    do ii=1,natm
                    !        !* Could convert this to lapack using my ddot wrapper
                    !        tmp = buffer_radial_g(ii,nn)*buffer_spherical_harm(ii,mm,ll)
                    !        
                    !        res(1) = res(1) + tmp
                    !        ! NOT SURE if conjg(sum x) = sum conjg(x)
                    !        res(2) = res(2) + dconjg(tmp)
                    !    end do
                    !
                    !    buffer_cnlm(mm,ll,nn) = res(1)

                    !    !* Y_-m,l = Y_ml^* * (-1)^m
                    !    buffer_cnlm(-mm,ll,nn) = res(2) * (-1.0d0**mm)
                    !end do
                end do
            end do
        end subroutine calc_cnlm

        subroutine init_buffer_factorial()
            implicit none

            integer :: ll,lmax

            lmax = bispect_param%lmax

            if(allocated(buffer_factorial)) then
                deallocate(buffer_factorial)
            end if
            allocate(buffer_factorial(0:3*lmax+1))
        
            buffer_factorial(0) = 1.0d0
            do ll=1,3*lmax+1
                buffer_factorial(ll) = dble(ll)*buffer_factorial(ll-1)
            end do
        end subroutine init_buffer_factorial

end module features
