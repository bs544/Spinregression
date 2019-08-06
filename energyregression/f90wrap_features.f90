! Module features defined in file features.f90

subroutine f90wrap_calculate_local(cell, atom_positions, rcut, weightings, &
    parallel, nmax, calc_type, buffer_size, x, n0, n1, n2, n3, n4, n5, n6)
    use features, only: calculate_local
    implicit none
    
    real(8), intent(in), dimension(n0,n1) :: cell
    real(8), intent(in), dimension(n2,n3) :: atom_positions
    real(8), intent(in) :: rcut
    real(8), intent(in), dimension(n4) :: weightings
    logical, intent(in) :: parallel
    integer, intent(in) :: nmax
    integer, intent(in) :: calc_type
    integer, intent(in) :: buffer_size
    real(8), intent(inout), dimension(n5,n6) :: x
    integer :: n0
    !f2py intent(hide), depend(cell) :: n0 = shape(cell,0)
    integer :: n1
    !f2py intent(hide), depend(cell) :: n1 = shape(cell,1)
    integer :: n2
    !f2py intent(hide), depend(atom_positions) :: n2 = shape(atom_positions,0)
    integer :: n3
    !f2py intent(hide), depend(atom_positions) :: n3 = shape(atom_positions,1)
    integer :: n4
    !f2py intent(hide), depend(weightings) :: n4 = shape(weightings,0)
    integer :: n5
    !f2py intent(hide), depend(x) :: n5 = shape(x,0)
    integer :: n6
    !f2py intent(hide), depend(x) :: n6 = shape(x,1)
    call calculate_local(cell=cell, atom_positions=atom_positions, rcut=rcut, &
        weightings=weightings, parallel=parallel, nmax=nmax, calc_type=calc_type, &
        buffer_size=buffer_size, X=x)
end subroutine f90wrap_calculate_local

! End of module features defined in file features.f90

