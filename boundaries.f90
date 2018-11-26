module boundaries
    use config
    use io

    implicit none
    
    !* linear algebra routines
    external :: dgemv    

    integer,private :: buffer_size1=1000

    contains
        subroutine find_neighbouring_images(neighbouring_images,buffer_size)
            ! find all cell images n : gamma = gamma + n such that
            ! |r| = |gamma*L + n*L| <= rcut
            !
            ! cartesians_xn = sum_i cell_xi * fractional_in
            !
            ! Arguments
            ! ---------
            ! neighbouring_images, shape=(,)
            !
            ! Notes
            ! -----
            ! assume atoms are located along all faces of local cell

            implicit none
    
            real(8),external :: dnrm2

            ! args
            real(8),allocatable,intent(inout) :: neighbouring_images(:,:)
            integer,intent(in),optional :: buffer_size

            !* scratch
            real(8) :: scratch_images(1:3,1:buffer_size1)
            integer :: n1,n2,n3,cntr,lim
            real(8) :: dn(1:3),dr_vec(1:3),dr
            real(8) :: dble_nvec(1:3)

            if (present(buffer_size)) then
                buffer_size1 = buffer_size
            end if

            !* For stupendously large rcut, this will need to be larger. No warning will be given, just a capped interaction V
            !lim=100
            lim=10

            dr_vec = 0.0d0        
 
            cntr = 0 
            do n1=-lim,lim,1
                dn(1) = dble(adjust_boundary(n1))
                dble_nvec(1) = dble(n1)
                do n2=-lim,lim,1
                    dn(2) = dble(adjust_boundary(n2))
                    dble_nvec(2) = dble(n2)
                    do n3=-lim,lim,1
                        dn(3) = dble(adjust_boundary(n3))
                        dble_nvec(3) = dble(n3)

                        !* displacement in cartesians
                        call dgemv('n',3,3,1.0d0,structure%cell,3,dn,1,0.0d0,dr_vec,1)
                        
                        !* displacement magniture (cartesians)
                        dr = dnrm2(3,dr_vec,1)

                        if (dr.le.bispect_param%rcut) then
                            cntr = cntr +1
                            scratch_images(:,cntr) = dble_nvec(:)
            
                            if (cntr+1.gt.buffer_size1) then
                                call error_message("find_neighbouring_images",&
                                    &"increase buffer_size1")
                            end if
                        end if
                    end do
                end do
            end do 
            !* (3,num_images)
            allocate(neighbouring_images(3,cntr))
            neighbouring_images(:,:) = scratch_images(:,1:cntr) 
        end subroutine find_neighbouring_images


        integer function adjust_boundary(n)
            implicit none

            integer,intent(in) :: n

            integer :: res

            if (n.eq.0) then
                res = 0
            else if (n.gt.0) then
                res = n - 1
            else if (n.lt.0) then
                res = n + 1
            end if
            adjust_boundary = res
        end function adjust_boundary

end module boundaries
