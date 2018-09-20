! l should be from 0!

module features
    use config
    use io, only :: error_message

    real(8),external :: ddot

    implicit none

    contains
        integer function cardinality_bispectrum_type1(lmax,nmax)
            implicit none

            integer,intent(in) :: lmax,nmax

            !* l=[0,lmax] , n=[1,nmax]
            cardinality_bispectrum_type1 = (lmax+1)*nmax
        end function cardinality_bispectrum_type1

        subroutine calculate_bispectrum_type1(cell,atom_positions,grid_coordinates,rcut,parallel,&
        &lmax,nmax,X)
            ! Compute bispectrum features as in [1]
            !
            ! x_nl = sum_{m=-l}^{m=l} c_{nlm}^* c_{nlm} 
            !
            ! Arguments
            ! ---------
            ! cell             : shape=(3,3)
            ! atom_positions   : shape=(3,Natm)
            ! grid_coordinates : shape=(3,Ngrid)
            !
            ! Note
            ! ----
            ! This routine uses full periodic image convention, not nearest 
            ! image
            !
            ! [1] PHYSICAL REVIEW B 87, 184115 (2013)

            implicit none

            !* args
            real(8),intent(in) :: cell(1:3,1:3),atom_positions(:,:)
            real(8),intent(in) :: grid_coordinates(:,:),rcut
            logical,intent(in) :: parallel
            integer,intent(in) :: lmax,nmax
            real(8),intent(inout) :: X(:,:)

            !* scratch
            integer :: dim(1:2),natm,ngrid,ii,loop(1:2)
            real(8),allocatable :: neigh_images(:,:),polar(:,:)

            !===============!
            !* arg parsing *!
            !===============!

            dim=shape(atom_positions)
        
            !* number of atoms in unit cell
            natm = dim(2)

            dim = shape(grid_coordinates)
        
            !* number of  density points to calculate features for
            ngrid = dim(2)

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

                    !* get type1 features
                    call features_bispectrum_type1(polar,X(:,ii)) 
                end do
            else
                write(*,*) "Not implemented"
                call exit(0)
            end if            
        end subroutine calculate_bispectrum_type1

        subroutine features_bispectrum_type1(polar,x)
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
            integer :: Nneigh,ll,nn,ii
            real(8) :: val_ln,reduce_array(1:2),tmp1,tmp2(1:2)

            dim = shape(polar)

            !* number of neighbouring atoms to grid point
            Nneigh = dim(1)

            !* redundancy arrays specific to grid point
            call init_buffer_all_polar(polar)

            cntr = 1
            do ll=0,bispect_param%lmax,1
                do nn=1,bispect_param%nmax,1    
                    do mm=1,ll 
                        reduce_array = 0.0d0
                        
                        neighbour_loop : do ii=1,Nneigh,1
                            tmp1 = buffer_radial_g(ii,nn)*buffer_spherical_p(ii,mm,ll)

                            tmp2(1) = tmp1*buffer_polar_sc(1,ii)
                            tmp2(2) = tmp1*buffer_polar_sc(2,ii)

                            reduce_array = reduce_array + tmp2
                        end do neighbour_loop

                        val_ln = val_ln + spherical_harm_const(ll,mm)*sum(reduce_array**2)
                    end do
                    
                    !* count -,+mm
                    val_ln = val_ln*2.0d0

                    !* m=0 contribution : cos(m phi)=1,sin(m phi)=0
                    val_ln = val_ln + ddot(Nneigh,buffer_radial_g(:,nn),1,buffer_spherical_p(:,0,ll))**2 * &
                    &spherical_harm_const(ll,0)

                    x(cntr) = val_ln
                    cntr = cntr + 1
                end do
            end do
        end subroutine features_bispectrum_type1

        real(8) function spherical_harm_const(l,m)
            ! l = [1,lmax]
            ! m = [-lmax,lmax]
            !
            ! store in array array = [midx,lidx]
            ! lidx = l, midx = m + lmax + 1
            !
            ! lidx = [1,lmax], midx=[1,2*lmax+1]
        
            implicit none
            
            !* args
            integer,intent(in) :: l,m

            !* scratch
            integer :: midx

            !* use array indices >= 1
            midx = spherical_harm_const__sub1(m)

            spherical_harm_const = buffer_spherical_harm_const(midx,l)
        end function spherical_harm_const

        integer function spherical_harm_const__sub1(m)
            implicit none

            !* args
            integer,intent(in) :: m

            !* map to interval [1,2*lmax+1]
            spherical_harm_const__sub1 = m + bispect_param%lmax + 1
        end function spherical_harm_const__sub1

        subroutine init_buffer_all_general()
            implicit none

            call init_buffer_spherical_harm_const()
            call init_buffer_radial_phi_Nalpha()
        end subroutine init_buffer_all_general

        subroutine init_buffer_spherical_harm_const()
            implicit none

            !* scratch
            integer :: ll,mm,midx

            if(allocated(buffer_spherical_harm_const)) then
                deallocate(buffer_spherical_harm_const)
            end if
            
            allocate(buffer_spherical_harm_const(1:bispect_param%lmax*2+1,bispect_param%lmax))

            do ll=1,bispect_param%lmax,1
                do mm=-ll,ll,1
                    midx = spherical_harm_const__sub1(mm)

                    buffer_spherical_harm_const(midx,ll) = spherical_harm_const__sub2(ll,mm)
                end do
            end do
        end subroutine init_buffer_spherical_harm_const

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

        subroutine init_buffer_radial_w()
            implicit none

            !* scratch
            real(8),allocatable :: S(:,:),ii,jj

            allocate(S(bispect_param%nmax,bispect_param%nmax))
            if (allocated(buffer_radial_w)) then
                deallocate(buffer_radial_w)
            end if
            allocate(buffer_radial_w(bispect_param%nmax,bispect_param%nmax))

            do ii=1,bispect_param%nmax,1
                do jj=1,bispect_param%nmax,1
                    S(ii,jj) = sqrt( (5.0d0+2.0d0*dble(ii))*(5.0d0+2.0d0*dble(jj)) ) / (5.0d0 + dble(ii) + dble(jj))
                end do
            end do

            !* W = S^{-1/2}
            buffer_radial_w = 0.0d0 ! THIS IS WRONG
        end subroutine init_buffer_radial_w

        subroutine init_buffer_spherical_p(polar)
            ! calculate associated legendre polynomail for cos(theta_j) for all
            ! atoms j
            use spherical_harmonics, only : plgndr

            implicit none

            !* args
            real(8),intent(in) :: polar(:,:)

            !* scratch
            integer :: dim(1:2)
            integer :: Nneigh,mm,ll,ii

            !* number of neighbouring atoms
            dim = shape(dim)
            Nneigh = dim(2)

            if(allocated(buffer_spherical_polar)) then
                deallocate(buffer_spherical_polar)
            end if
            allocate(buffer_spherical_polar(1:Nneigh,0:bispect_param%lmax,0:bispect_param%lmax)

            !*              m       l
            !* [1,Nneigh][0,lmax][0,lmax]
            buffer_spherical_polar = 0.0d0

            do ll=0,bispect_param%lmax,1
                do mm=0,ll,1
                    do ii=1,Nneigh,1
                        buffer_spherical_p(ii,mm,ll) = plgndr(ll,mm,cos(polar(2,ii)))
                    end do
                end do
            end do
        end subroutine init_buffer_spherical_p

        subroutine init_buffer_polar_sc(polar)
            implicit none

            !* args
            real(8),intent(in) :: polar(:)

            !* scratch
            integer :: dim(1:2),Nneigh,ii

            dim = shape(polar)

            Nneigh = dim(2)

            if(allocated(buffer_polar_sc)) then
                deallocate(buffer_polar_sc)
            end if
            allocate(buffer_polar_sc((1:2,1:Nneigh)))
            
            do ii=1,Nneigh,1
                !* (cos(phi),sin(phi))
                buffer_polar_sc(1,ii) = cos(polar(3,ii))
                buffer_polar_sc(2,ii) = sin(polar(3,ii))
            end do
        end subroutine init_buffer_polar_sc

        subroutine init_buffer_all_polar(polar)
            ! initialise and compute all redundancy arrays for info specific
            ! to a particular density grid point
            implicit none

            real(8),intent(in) :: polar(:,:)            

            !* associated legendre polynomials of polar angle
            call init_buffer_spherical_p(polar)

            !* trigonometric bases for azimuthal angle
            call init_buffer_polar_sc(polar)
        
            !* radial component in radial bases
            call init_radial_g(polar)
        end subroutine init_buffer_all_polar

        real(8) function spherical_harm_const__sub2(ll,mm)
            implicit none
    
            !* args
            integer,intent(in) :: ll,mm

            !* scratch
            real(8) :: dble_ll,dble_mm

            dble_ll = dble(ll)
            dble_mm = dble(mm)
        
            spherical_harm_const__sub2 = ((2.0d0*dble_ll+1.0d0)*factorial(ll-mm))/(12.5663706144d0*factorial(ll+mm))
        end function spherical_harm_const__sub2

        integer recursive function factorial(x) result(res)
            implicit none

            !* args
            integer,intent(in) :: x
    
            if (x.eq.0) then
                res = 1
            else if (x.gt.0) then
                res = x*factoriral(x-1)
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

            allocate(phi(bispect_param%nmax,Nneigh))
            if (allocated(buffer_radial_g)) then
                deallocate(buffer_radial_g)
            end if
            allocate(buffer_radial_g(Nneigh,bispect_param%nmax))

            do nn=1,bispect%param%nmax,1
                do ii=1,Nneigh,1
                    phi(ii,nn) = radial_phi_type1(nn,polar(1,ii))
                end do
            end do        

            !* g_ni = sum_alpha W_n,alpha phi_alpha,i
            call dgemm('n','n',bispect_param%nmax,Nneigh,bispect_param%nmax,1.0d0,buffer_radial_w,bispect_param%nmax,&
            &phi,bispect_param%nmax,0.0d0,buffer_radial_g,bispect_param%nmax)
        end subroutine init_radial_g

end module features
