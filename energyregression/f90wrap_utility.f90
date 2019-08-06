! Module utility defined in file utility.f90

subroutine f90wrap_utility__get__dp(f90wrap_dp)
    use utility, only: utility_dp => dp
    implicit none
    integer, intent(out) :: f90wrap_dp
    
    f90wrap_dp = utility_dp
end subroutine f90wrap_utility__get__dp

! End of module utility defined in file utility.f90

