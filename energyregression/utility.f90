module utility
    use io

    implicit none

    external :: dsyev
    external :: dgemv
    external :: dgetrf
    external :: dgetri

    integer,parameter :: dp=selected_real_kind(15,300)

    contains

        subroutine sqrt_of_matrix_inverse(symmetric_matrix,invsqrt)
            ! perform inverse via eigen decomposition
            ! B = V D V^{-1} where D is diagonal matrix of eigenvalues, then
            ! B^{1/2} = V D^{1/2} V^{-1}
            !
            ! Likewise, B^{-1} = V D^{-1} V ^{-1}  since
            ! (ABC)^{-1} = C^{-1}B^{-1}A^{-1} and since D^{-1} is still diagonal
            ! , B^{-1/2} = V D^{-1/2} V^{-1} where D^{-1/2}_ii = 1/sqrt(D_ii)
            !
            ! So if S=B^{-1/2}, then S = V D^{-1/2} V^{-1} where B = V D V^{-1}
            !
            ! Note V is constructed of eigenvectors of B, D is constructed of
            ! eigenvalues of B
            implicit none

            !* args
            real(8),intent(in) :: symmetric_matrix(:,:)
            real(8),intent(inout) :: invsqrt(:,:)

            !* scratch
            real(8),allocatable,dimension(:,:) :: V,D,invV
            real(8),allocatable,dimension(:,:) :: tmp1,tmp2
            integer :: ii,dim(1:2)

            !* A = V D V^{-1} = V D V^T since V^{-1}=V^T when A is symmetric
            call symmetric_eigen_decomposition(symmetric_matrix,V,D,invV)

            dim = shape(symmetric_matrix)
            do ii=1,dim(1)
                if (D(ii,ii).lt.0.0d0) then
                    write(*,*) "Error in sqrt_of_matrix_inverse : negative eigenvalue ",D(ii,ii),"computed",&
                    &"matrix must be positive definite (no negative eigenvalues"
                    call exit(0)
                end if

                !* A^{-1/2} = V D^{-1/2} V^^{-1} : D^{-1/2}_ii = 1.0/sqrt(D_ii)
                D(ii,ii) = 1.0/sqrt(D(ii,ii))
            end do

            allocate(tmp1(dim(1),dim(2)))
            allocate(tmp2(dim(1),dim(2)))


            !* tmp1 = D^{-1/2} V^{-1} = D^{-1/2} V^T
            call dgemm('n','n',dim(1),dim(2),dim(2),1.0d0,D,dim(1),invV,dim(2),0.0d0,tmp1,dim(1))

            !* invsqrt = V D^{-1/2} V^{-1}
            call dgemm('n','n',dim(1),dim(2),dim(2),1.0d0,V,dim(1),tmp1,dim(2),0.0d0,invsqrt,dim(1))
        end subroutine sqrt_of_matrix_inverse

        subroutine eigen_vectors_values(symmetric_matrix,eigenvalues,eigenvectors)
            ! when A is symmetric, A = Q T Q^{H} where T is called real
            ! tridiagonal form of A.

            implicit none

            !external :: dsyev_

            !* args
            real(8),intent(in) :: symmetric_matrix(:,:)
            real(8),allocatable,intent(inout) :: eigenvalues(:),eigenvectors(:,:)

            !* scratch
            integer :: dim(1:2)
            integer :: ii,jj
            real(8) :: atol
            real(8),allocatable :: work_array(:)
            integer :: dim_work,info

            dim = shape(symmetric_matrix)

            if (dim(1).ne.dim(2)) then
                write(*,*) "Error in eigen_vectors_values : matrix not square"
                call exit(0)
            end if
            !* check matrix is transpose
            do ii=1,dim(1),1
                do jj=1,dim(2),1
                    !* get absolute tol
                    if (abs(symmetric_matrix(ii,jj)).lt.dble(1e-15)**2) then
                        atol = dble(1e-12)**2
                    else
                        atol = abs(symmetric_matrix(ii,jj))*dble(1e-9)
                    end if

                    if (abs(symmetric_matrix(ii,jj)-symmetric_matrix(jj,ii)).gt.atol) then
                        write(*,*) "Error in eigen_vectors_values : matrix not symmetric : ",&
                        &symmetric_matrix(ii,jj),"!=",symmetric_matrix(jj,ii)
                        call exit(0)
                    end if
                end do
            end do

            allocate(eigenvalues(dim(1)))
            allocate(eigenvectors(dim(1),dim(2)))

            !* copy to working array
            eigenvectors(:,:) = symmetric_matrix(:,:)

            !* length of work array
            dim_work = 3*dim(1) - 1

            !* work array
            allocate(work_array(dim_work))

            !* return eigenvectors in A
            info = 0

            !* eigenvalues(coordinate, eigen vector number)
            call dsyev('V','U',dim(1),eigenvectors,dim(1),eigenvalues,work_array,dim_work,info)

            if (info.ne.0) then
                write(*,*) "Error in eigen_vectors_values : failed to find eigenvectors/values"
                call exit(0)
            end if
        end subroutine eigen_vectors_values

        logical function check_eigen_vectors_values(matrix,eig_values,eig_vectors,verbose)
            implicit none

            !* args
            real(8),intent(in) :: matrix(:,:),eig_values(:),eig_vectors(:,:)
            logical,optional,intent(in) :: verbose

            !* scratch
            integer :: dim(1:2),ii,jj
            real(8),allocatable :: tmp(:),tmp2(:)
            real(8) :: atol
            logical :: res=.true.


            dim = shape(matrix)
            allocate(tmp(dim(2)))
            allocate(tmp2(dim(2)))

            do ii=1,dim(1)
                !* LHS
                call dgemv('n',dim(1),dim(2),1.0d0,matrix,dim(1),eig_vectors(:,ii),1,0.0d0,tmp,1)

                !* RHS
                tmp2 = eig_values(ii)*eig_vectors(:,ii)

                do jj=1,dim(2),1
                    if (abs(tmp2(jj)).lt.dble(1e-15)**2) then
                        atol = dble(1e-12)**2
                    else
                        atol = abs(tmp2(jj))*dble(1e-9)
                    end if

                    if (abs(tmp(jj)-tmp2(jj)).gt.atol) then
                        res = .false.
                        if (present(verbose)) then
                            if (verbose) then
                                write(*,*) "failing ",tmp(jj),tmp2(jj),abs(tmp(jj)-tmp2(jj))
                            end if
                        end if
                    end if
                end do
            end do
            check_eigen_vectors_values = res
        end function check_eigen_vectors_values

        subroutine symmetric_eigen_decomposition(symmetric_matrix,V,D,invV)
            ! perform eigen decomposition for symmetric matrix by constructing
            ! diagonals of eigenvalues D and matrix of eigenvectors V such that
            ! V_ij = ith component of jth eigenvector
            !
            ! A = V D V^{-1} but when A is symmetric, V^{-1}=V^T, so
            ! A = V D V^T

            implicit none

            real(8),intent(in) :: symmetric_matrix(:,:)
            real(8),allocatable,intent(inout) :: V(:,:),D(:,:),invV(:,:)

            !* scratch
            real(8),allocatable :: eig_vals(:)
            integer :: dim(1:2),ii,jj

            call eigen_vectors_values(symmetric_matrix,eig_vals,V)

            dim = shape(V)
            allocate(D(dim(1),dim(2)))
            allocate(invV(dim(1),dim(2)))

            D = 0.0d0
            do ii=1,dim(1)
                D(ii,ii) = eig_vals(ii)
            end do

            !* V^{-1} = V^T when A is symmetric
            do ii=1,dim(1)
                do jj=1,dim(2)
                    invV(ii,jj) = V(jj,ii)
                end do
            end do
        end subroutine symmetric_eigen_decomposition

        logical function check_symmetric_eigen_decomposition(symmetric_matrix,V,D,invV,verbose)
            ! check that A = V D V^{-1}
            ! When A is symmetric, V^{-1}=V^T

            implicit none

            !* args
            real(8),intent(in) :: symmetric_matrix(:,:),V(:,:),D(:,:),invV(:,:)
            logical,optional,intent(in) :: verbose

            !* scratch
            real(8),allocatable :: tmp2(:,:),tmp3(:,:)
            integer :: dim(1:2),ii,jj
            logical :: res=.true.
            real(8) :: atol

            dim = shape(symmetric_matrix)

            allocate(tmp2(dim(1),dim(2)))
            allocate(tmp3(dim(1),dim(2)))

            !* tmp2 = D V^T
            call dgemm('n','n',dim(1),dim(2),dim(2),1.0d0,D,dim(1),invV,dim(2),0.0d0,tmp2,dim(1))

            !* tmp3 = V tmp2 = V D V^T
            call dgemm('n','n',dim(1),dim(2),dim(2),1.0d0,V,dim(1),tmp2,dim(2),0.0d0,tmp3,dim(1))

            !* check that A = V D V^T
            do ii=1,dim(1)
                do jj=1,dim(2)
                    if (abs(symmetric_matrix(ii,jj)).lt.dble(1e-15)**2) then
                        atol = dble(1e-12)**2
                    else
                        atol = abs(symmetric_matrix(ii,jj))*dble(1e-9)
                    end if

                    if (abs(symmetric_matrix(ii,jj)-tmp3(ii,jj)).gt.atol) then
                        if (present(verbose)) then
                            if (verbose) then
                                write(*,*) "Error in check_symmetric_eigen_decomposition : Decomposition failed with ",&
                                &symmetric_matrix(ii,jj),tmp3(ii,jj)
                            end if
                        end if
                        res = .false.
                    end if
                end do
            end do
            check_symmetric_eigen_decomposition = res
        end function check_symmetric_eigen_decomposition

        logical function check_sqrt_inv_matrix(symmetric_matrix,invsqrt,verbose)
            ! check that invsqrt*invsqrt = symmetric_matrix^{-1}
            implicit none

            !* args
            real(8),intent(in),dimension(:,:) :: symmetric_matrix,invsqrt
            logical,optional,intent(in) :: verbose

            !* scratch
            integer :: dim(1:2),info,ii,jj
            real(8),allocatable,dimension(:,:) :: matrix_inv_lhs,matrix_inv_rhs
            integer,allocatable :: ipiv(:)
            real(8),allocatable :: work(:)
            real(8) :: atol
            logical :: res=.true.

            dim = shape(symmetric_matrix)
            allocate(matrix_inv_lhs(dim(1),dim(2)))
            allocate(matrix_inv_rhs(dim(1),dim(2)))
            allocate(ipiv(dim(1)))
            allocate(work(dim(1)))

            matrix_inv_lhs(:,:) = symmetric_matrix(:,:)

            !* do easy part first, rhs = invsrqt . invsqrt
            call dgemm('n','n',dim(1),dim(2),dim(2),1.0d0,invsqrt,dim(1),invsqrt,dim(2),0.0d0,matrix_inv_rhs,dim(1))

            !* invert
            call dgetrf(dim(1),dim(1),matrix_inv_lhs,dim(1),ipiv,info)
            call dgetri(dim(1),matrix_inv_lhs,dim(1),ipiv,work,dim(1),info)
            if (info.ne.0) then
                write(*,*) "Error : matrix inversion in check_sqrt_inv_matrix failed"
                call exit(0)
            end if

            do ii=1,dim(1)
                do jj=1,dim(1)
                    if (abs(matrix_inv_lhs(ii,jj)).lt.dble(1e-15)**2) then
                        atol = dble(1e-12)**2
                    else
                        atol = abs(matrix_inv_lhs(ii,jj))*dble(1e-9)
                    end if

                    if (abs(matrix_inv_lhs(ii,jj)-matrix_inv_rhs(ii,jj)).gt.atol) then
                        res = .false.
                        if (present(verbose)) then
                            if (verbose) then
                                write(*,*) "Failing equality in check_sqrt_inv_matrix of ",&
                                &matrix_inv_lhs(ii,jj),matrix_inv_rhs(ii,jj)
                            end if
                        end if
                    end if
                end do
            end do
            check_sqrt_inv_matrix = res
        end function check_sqrt_inv_matrix

        subroutine load_balance_alg_1(thread_idx,num_threads,N,bounds)
            implicit none

            integer,intent(in) :: thread_idx,num_threads,N
            integer,intent(inout) :: bounds(1:2)

            !* scratch
            integer :: dx,all_bounds(1:2,1:num_threads),difference,thread

            !* average width of loop
            dx = int(floor(float(N)/float(num_threads)))

            !* remaining elements for final thread
            difference = N - dx*num_threads

            if ((difference.gt.num_threads).or.(thread_idx.lt.0).or.(thread_idx.ge.num_threads)) then
                call error_message("load_balance_alg_1","Check args to routine")
            end if

            all_bounds(1,1) = 1
            all_bounds(2,1) = all_bounds(1,1) + dx - 1
            if (1.le.difference) then
                all_bounds(2,1) = all_bounds(2,1) + 1
            end if

            do thread=2,num_threads
                all_bounds(1,thread) = all_bounds(2,thread-1) + 1
                all_bounds(2,thread) = all_bounds(1,thread) + dx - 1
                if (thread.le.difference) then
                    all_bounds(2,thread) = all_bounds(2,thread) + 1
                end if
            end do
            bounds(1:2) = all_bounds(1:2,thread_idx+1)
        end subroutine load_balance_alg_1

        subroutine sort_array(array)
            !sorts the array, ordered from lowest to highest
            implicit none
            integer, intent(inout) :: array(:)
            integer :: arr_len, ii, jj
            integer :: buffer

            arr_len = size(array)

            do ii = 1,arr_len
                jj = minloc(array(ii:arr_len),1)
                jj = jj + ii - 1
                buffer = array(ii)
                array(ii) = array(jj)
                array(jj) = buffer
            end do


        end subroutine sort_array

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
end module utility
