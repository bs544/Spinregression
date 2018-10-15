module spherical_harmonics
    use config
    
    implicit none

    contains
        real(8) function plgndr_dep(l,m,x)
            INTEGER l,m
            REAL(8) x
            INTEGER i,ll
            REAL(8) fact,oldfact,pll,pmm,pmmp1,omx2,PI
            PARAMETER(PI=3.1415927d0)
            
            if(m.lt.0.or.m.gt.l.or.abs(x).gt.1.) then
                write(*,*) 'bad arguments in plgndr'
                call exit(0)
            end if            

            pmm=1.
            
            if(m.gt.0) then
                omx2=(1.-x)*(1.+x)
                fact=1.
                do i=1,m
                    pmm=pmm*omx2*fact/(fact+1.)
                    fact=fact+2.
                enddo
            endif
            
            pmm=sqrt((2*m+1)*pmm/(4.*PI))
            
            if(mod(m,2).eq.1) pmm=-pmm
            
            if(l.eq.m) then
                plgndr_dep=pmm
            else
                pmmp1=x*sqrt(2.*m+3.)*pmm

                if(l.eq.m+1) then
                    plgndr_dep=pmmp1
                else
                    oldfact=sqrt(2.*m+3.)
                    do ll=m+2,l
                        fact=sqrt((4.*ll**2-1.)/(ll**2-m**2))
                        pll=(x*pmmp1-pmm/oldfact)*fact
                        oldfact=fact
                        pmm=pmmp1
                        pmmp1=pll
                    enddo
                    plgndr_dep=pll
                endif
            endif
            return
        end function plgndr_dep

        real(8) function plgndr_s(l,m,x)
            ! x must lie in range [-1,1]
            ! Numerical recipes for fortran
            implicit none

            integer,intent(in) :: l,m
            real(8),intent(in) :: x

            integer :: ll
            real(8) :: pll,pmm,pmmp1,somx2

            pmm=1.0d0
            if(m.gt.0) then
                somx2=sqrt((1.0d0-x)*(1.0d0+x))
                pmm=product(arth_d(1.0d0,2.0d0,m))*somx2**m
                if (mod(m,2).eq.1) pmm=-pmm
            end if
            if(l.eq.m) then
                plgndr_s = pmm
            else
                pmmp1=x*dble(2*m+1)*pmm
                if(l.eq.m+1) then
                    plgndr_s=pmmp1
                else
                    do ll=m+2,l
                        pll=(x*dble(2*ll-1)*pmmp1 - dble(ll+m-1)*pmm )/dble(ll-m)
                        pmm=pmmp1
                        pmmp1=pll
                    end do
                    plgndr_s=pll
                end if
            end if
        end function plgndr_s

        function arth_d(first,increment,n)
            ! numerical recipes nrutil, function arth_d

            real(8), intent(in) :: first,increment
            integer, intent(in) :: n
            
            real(8), dimension(n) :: arth_d
            integer :: k,k2,npar_arth,npar2_arth
            real(8) :: temp

            npar_arth = 16
            npar2_arth = 8

            if (n > 0) arth_d(1)=first
            if (n <= npar_arth) then
                do k=2,n
                    arth_d(k)=arth_d(k-1)+increment
                end do
            else
                do k=2,npar2_arth
                    arth_d(k)=arth_d(k-1)+increment
                end do
                temp=increment*npar2_arth
                k=NPAR2_ARTH
                do
                    if (k >= n) exit
                    k2=k+k
                    arth_d(k+1:min(k2,n))=temp+arth_d(1:min(k,n-k))
                    temp=temp+temp
                    k=k2
                end do
            end if
        end function arth_d

        real(8) function plgndr_debug(l,m,x)
            implicit none
            
            !* args
            real(8),intent(in) :: x
            integer,intent(in) :: l,m

       
            !* scratch
            real(8) :: res
           
            if ((m.lt.0).or.(m.gt.l)) then
                write(*,*) "0 <= m <= l is not true in plgndr_debug"
                call exit(0)
            end if

            if (l.eq.0) then
                if (m.eq.0) then
                    res = 1.0d0
                end if
            else if (l.eq.1) then
                if (m.eq.0) then
                    res = x
                else if (m.eq.1) then
                    res = -1.0d0*sqrt(1.0d0 - x**2)
                end if
            else if (l.eq.2) then
                if (m.eq.0) then
                    res = 0.5d0*(3.0d0*x**2 - 1.0d0)
                else if (m.eq.1) then
                    res = -3.0d0*x*sqrt(1.0d0-x**2)
                else if (m.eq.2) then
                    res = 3.0d0*(1.0d0-x**2)
                end if
            else if (l.eq.3) then
                if (m.eq.0) then
                    res = 0.5d0*x*(5.0d0*x**2-3.0d0)
                else if (m.eq.1) then
                    res = 1.5d0*(1.0d0-5.0d0*x**2)*sqrt(1.0d0-x**2)
                else if (m.eq.2) then
                    res = 15.0d0*x*(1.0d0-x**2)
                else if (m.eq.3) then
                    res = -15.0d0*(1.0d0-x**2)**(1.5d0)
                end if
            else if (l.eq.4) then
                if (m.eq.0) then
                    res = 1.0d0/8.0d0*(35.0d0*x**4 - 30.0d0*x**2 + 3.0d0)
                else if (m.eq.1) then
                    res = 5.0d0/2.0d0*x*(3.0d0-7.0d0*x**2)*sqrt(1.0d0-x**2)
                else if (m.eq.2) then
                    res = 15.0d0/2.0d0*(7.0d0*x**2-1.0d0)*(1.0d0-x**2)
                else if (m.eq.3) then
                    res = -105.0d0*x*(1.0d0-x**2)**(1.5d0)
                else if (m.eq.4) then
                    res = 105.0d0*(1.0d0-x**2)**2
                end if
            else
                write(*,*) "only values for l <= 4 are supported in plgndr_debug"
                call exit(0)
            end if
            plgndr_debug = res
        end function plgndr_debug

        complex(8) function sph_harm(m,l,theta,phi)
            !* theta,phi are inclination,azimuth angle respectively
            implicit none

            integer,intent(in) :: m,l
            real(8) :: theta,phi

            !* scratch
            real(8) :: klm,plm
            complex(8) :: tmp
    
            !* constant
            klm = sqrt(((2.0d0*dble(l)+1.0d0)*dble(factorial_duplicated(l-m)))/(12.5663706144d0*dble(factorial_duplicated(l+m))))

            !* associated legendre polynomial
            if (m.lt.0) then
                plm = (-1.0d0)**(-m)*plgndr_s(l,-m,cos(theta)) * dble(factorial_duplicated(l+m))/dble(factorial_duplicated(l-m))
            else
                plm = plgndr_s(l,m,cos(theta))
            end if

            !* exp(i phi m)
            tmp = complex(cos(phi*m),sin(phi*m)) * klm * plm
            sph_harm = tmp
        end function sph_harm

        integer recursive function factorial_duplicated(x) result(res)
            implicit none

            integer,intent(in) :: x

            if (x.eq.0) then    
                res = 1
            else if (x.gt.0) then
                res = x*factorial_duplicated(x-1)
            end if
        end function

        subroutine check_plgndr_s()
            implicit none

            !* scratch
            integer :: lmax,ll,mm,ii
            real(8) :: res1,res2
            real(8) :: atol
            real(8),allocatable :: x(:)

            allocate(x(100))

            lmax=4
            do mm=1,size(x)
                x(mm) = dble(mm-1)/dble(size(x)-1)*2.0d0 - 1.0d0
            end do

            do ll=0,lmax
                do mm=0,ll
                    do ii=1,size(x)
                        res1 = plgndr_s(ll,mm,x(ii))
                        res2 = plgndr_debug(ll,mm,x(ii))

                        if(abs(res2).lt.1e-20) then
                            atol = dble(1e-20)
                        else
                            atol = abs(res2*dble(1e-8))
                        end if

                        if(abs(res1-res2).gt.atol) then
                            write(*,*) "Inequality in check_plgndr_s for (m,l)=",mm,ll
                            write(*,*) ""
                            write(*,*) res1,"!=",res2,": atol = ",atol,"difference = ",abs(res1-res2)
                            call exit(0)
                        end if
                    end do
                end do
            end do
        end subroutine check_plgndr_s

        real(8) function ql_bruteforce(f_radial,polar,l)
            ! compute ql = sum_{m=-l}^{m=l} |c_nlm|^2 where
            ! c_nlm = sum_i f_radial(ii) * Y_lm(ii)
            implicit none

            integer,intent(in) :: l
            real(8),intent(in) :: f_radial(:),polar(:,:)

            integer :: dim(1:1),ii,mm
            complex(8),allocatable :: c_nlm(:)
            real(8) :: theta,phi
            real(8),allocatable :: qml(:)

            dim = shape(f_radial)
            allocate(c_nlm(-l:l))
            allocate(qml(-l:l))


            do mm=-l,l
                c_nlm(mm) = complex(0.0d0,0.0d0)
                do ii=1,dim(1)
                    theta = polar(2,ii) ! inclination
                    phi   = polar(3,ii) ! azimth

                    c_nlm(mm) = c_nlm(mm) + f_radial(ii)*sph_harm(mm,l,theta,phi)
                end do
                qml(mm) = abs(c_nlm(mm))**2
            end do                
            ql_bruteforce = sum(qml)
        end function ql_bruteforce
        
        
        real(8) function cg_su2( j1t, j2t, j3t, m1t, m2t, m3t )
            !-------------------------------------------------------------------------
            !                             ****************
            !                             ***   DWR3   ***
            !                             ****************
            ! ----------------------------------------------------------------------
            ! Update:        ... original code by J.P. Draayer
            !          05/01 ... modified according to SU3CGVCS by C. Bahri
            ! ----------------------------------------------------------------------
            !     Wigner coefficients for SO(3) -- triangle relations checked in
            !     delta.
            ! ----------------------------------------------------------------------
            ! References:
            !  1. M.E. Rose, Elementary Theory of Angular Momentum (Wiley)
            ! ----------------------------------------------------------------------
            ! C. Bahri, D.J. Rowe, J.P. Draayer, Computer Physics Communications
            ! , 159, (2004), Eslevier
            !     ------------------------------------------------------------------
            implicit none
            
            !* args
            integer j1t, j2t, j3t, m1t, m2t, m3t

            ! scratch
            integer i1, i2, i3, i4, i5, it, itmin, itmax
            real(8) :: dc, dtop, dbot, dsum, dwr3
            real(8) :: dlogf(0:2000)

            !real(8) :: dexp, dlogf  
            !logical :: btest
            !common / BKDF / dlogf(0:2000)

            dwr3 = 0.d0
            if( m1t+m2t-m3t .ne. 0 ) then
                cg_su2 = dwr3 
            end if
            
            dc = dlogf(j1t+j2t-j3t) + dlogf(j2t+j3t-j1t) + dlogf(j3t+j1t-j2t) - dlogf(j1t+j2t+j3t+2)
            
            
            i1 = j3t - j2t + m1t
            i2 = j3t - j1t - m2t
            i3 = j1t + j2t - j3t
            i4 = j1t - m1t
            if( btest( i4, 0 ) ) then
                cg_su2 = dwr3
            end if
            i5 = j2t + m2t
            if( btest( i5, 0 ) ) then
                cg_su2 = dwr3
            end if

            itmin = max0( 0, -i1, -i2 )
            itmax = min0( i3, i4, i5 )
            if( itmin .gt. itmax ) then
                cg_su2 = dwr3
            end if

            dtop = ( dlog(dfloat(j3t+1)) + dc + dlogf(j1t+m1t) +&
                   &dlogf(j1t-m1t) + dlogf(j2t+m2t) + dlogf(j2t-m2t) +&
                   &dlogf(j3t+m3t) + dlogf(j3t-m3t) ) / dfloat(2)
            do it = itmin, itmax, 2
                dbot = dlogf(i3-it) + dlogf(i4-it) + dlogf(i5-it) +&
                       &dlogf(it) + dlogf(i1+it) + dlogf(i2+it)
                dsum = dexp(dtop-dbot)
                if( btest( it, 1 ) ) then
                  dwr3 = dwr3 - dsum
                else
                  dwr3 = dwr3 + dsum
                end if
            end do
            cg_su2 = dwr3
        end function cg_su2

       real(8) function cg_varshalovich(l_1,m_1,l_2,m_2,l,m)
            ! C_{l_1,m_1,l_2,m_2}^{l,m}
            ! where l,m values are int of half int
            implicit none

            real(8),intent(in) :: l_1,m_1,l_2,m_2,l,m

            !* scratch
            real(8) :: minimum,min_array(1:7),sqrtres
            real(8) :: imin,imax,val,sumres,sqrtarg
            real(8) :: dble_ii
            integer :: ii

            if (abs(m_1 + m_2 - m).gt.1e-15) then
                cg_varshalovich = 0.0d0
            else
                min_array(1) = l_1 + l_2 - l
                min_array(2) = l_1 - l_2 + l
                min_array(3) = -l_1 + l_2 + l
                min_array(4) = l_1 + l_2 + l + 1.0d0
                min_array(5) = l_1 - abs(m_1)
                min_array(6) = l_2 - abs(m_2)
                min_array(7) = l - abs(m)

                minimum = minval(min_array)

                if (minimum.lt.0.0d0) then
                    cg_varshalovich = 0.0d0
                else
                    ! NOTE : Python int(x)=floor(x) for x>0, int(x)=ceil(x) for x<0
                    ! NOTE : without casting to dble, int overflows for large l
                    sqrtarg = 1.0d0
                    sqrtarg = sqrtarg * buffer_factorial(python_int(l_1+m_1))
                    sqrtarg = sqrtarg * buffer_factorial(python_int(l_1-m_1))
                    sqrtarg = sqrtarg * buffer_factorial(python_int(l_2+m_2))
                    sqrtarg = sqrtarg * buffer_factorial(python_int(l_2-m_2))
                    sqrtarg = sqrtarg * buffer_factorial(python_int(l+m))
                    sqrtarg = sqrtarg * buffer_factorial(python_int(l-m))
                    sqrtarg = sqrtarg * dble((int(2.0d0*l) + 1))
                    sqrtarg = sqrtarg * buffer_factorial(python_int(min_array(1)))
                    sqrtarg = sqrtarg * buffer_factorial(python_int(min_array(2)))
                    sqrtarg = sqrtarg * buffer_factorial(python_int(min_array(3)))

                    ! sqrtarg is int so need to divide after casting to double
                    sqrtres = sqrt(sqrtarg / buffer_factorial(python_int(min_array(4))))
                    
                    min_array(1) = l_1 + m_2 - l
                    min_array(2) = l_2 - m_1 - l
                    min_array(3) = 0.0d0
                    min_array(4) = l_2 + m_2
                    min_array(5) = l_1 - m_1
                    min_array(6) = l_1 + l_2 - l

                    imin = maxval(min_array(1:3))
                    imax = minval(min_array(4:6))
                    sumres = 0.0d0
                    do ii=python_int(imin),python_int(imax)
                        dble_ii = dble(ii)
                        val = 1.0d0
                        val = val * buffer_factorial(ii)
                        val = val * buffer_factorial(python_int(l_1 + l_2 - l - dble_ii ))
                        val = val * buffer_factorial(python_int(l_1 - m_1 - dble_ii ))
                        val = val * buffer_factorial(python_int(l_2 + m_2 - dble_ii ))
                        val = val * buffer_factorial(python_int(l - l_2 + m_1 + dble_ii ))
                        val = val * buffer_factorial(python_int(l - l_1 - m_2 + dble_ii ))
                        sumres = sumres + (-1.0d0)**ii / val
                    end do
                    cg_varshalovich = sqrtres * sumres
                end if
            end if
        end function cg_varshalovich

        integer function python_int(x)
            ! python int(x) = floor(x) for x>0 , python int(x) = ceil(x), x<0
            implicit none

            real(8),intent(in) :: x

            if (x.ge.0.0d0) then
                python_int = int(floor(x))
            else
                python_int = int(ceiling(x))
            end if
        end function python_int
end module spherical_harmonics 
