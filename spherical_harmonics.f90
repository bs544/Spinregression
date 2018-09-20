module spherical_harmonics


    contains
        real(8) FUNCTION plgndr(l,m,x)
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
                plgndr=pmm
            else
                pmmp1=x*sqrt(2.*m+3.)*pmm

                if(l.eq.m+1) then
                    plgndr=pmmp1
                else
                    oldfact=sqrt(2.*m+3.)
                    do ll=m+2,l
                        fact=sqrt((4.*ll**2-1.)/(ll**2-m**2))
                        pll=(x*pmmp1-pmm/oldfact)*fact
                        oldfact=fact
                        pmm=pmmp1
                        pmmp1=pll
                    enddo
                    plgndr=pll
                endif
            endif
            return
        END

end module spherical_harmonics 
