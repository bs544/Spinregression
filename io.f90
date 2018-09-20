module io
    implicit none

    contains
        subroutine error_message(traceback,message)
            implicit none

            !* args        
            character(len=*),intent(in) :: traceback,message

            !* scratch
            character,allocatable :: header(:)

            allocate(header(len(traceback)+27))
            header = "*"
        
            write(*,*) ""
            write(*,*) header
            write(*,*) "Error raised in routine : ",traceback
            write(*,*) header
            write(*,*) ""
            write(*,*) "Message : ",message
            call exit(0)
        end subroutine error_message
end module io
