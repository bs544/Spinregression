module utility
    implicit none
    
    external :: dsyev
    external :: dgemv

    contains
        subroutine sqrt_of_matrix_inverse(symmetric_matrix)
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
    
            real(8),intent(in) :: symmetric_matrix 
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
                        write(*,*) "Error in eigen_vectors_values : matrix not symmetryic : ",&
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
            real(8),allocatable :: tmp(:),tmp2(:),atol
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
                write(*,*) tmp,eig_values(ii)*eig_vectors(:,ii)
            end do
            check_eigen_vectors_values = res
        end function check_eigen_vectors_values

        subroutine symmetric_eigen_decomposition(symmetric_matrix)
            ! perform eigen decomposition for symmetric matrix by constructing
            ! diagonals of eigenvalues D and matrix of eigenvectors V such that
            ! V_ij = ith component of jth eigenvector
            !
            ! A = V D V^{-1} but when A is symmetric, V^{-1}=V^T, so
            ! A = V D V^T

            implicit none

            real(8),intent(in) :: symmetric_matrix(:,:)

            !* scratch
            real(8),allocatable :: V(:,:),D(:,:),eig_vals(:)
            integer :: dim(1:2),ii

            call eigen_vectors_values(symmetric_matrix,eig_vals,V)

            dim = shape(V)
            allocate(D(dim(1),dim(2)))

            D = 0.0d0
            do ii=1,dim(1)
                D(ii,ii) = eig_vals(ii)
            end do 
        end subroutine symmetric_eigen_decomposition

        subroutine check_symmetric_eigen_decomposition(symmetric_matrix,V,D)
            ! check that A = V D V^{-1}                
            ! When A is symmetric, V^{-1}=V^T

            implicit none

            !* args
            real(8),intent(in) :: symmetric_matrix(:,:),V(:,:),D(:,:)

            !* scratch
            real(8),allocatable :: tmp1(:,:),tmp2(:,:),tmp3(:,:)
            integer :: dim(1:2),ii,jj
            logical :: res=.true.

            dim = shape(symmetric_matrix)

            allocate(tmp1(dim(1),dim(2)))
            allocate(tmp2(dim(1),dim(2)))
            allocate(tmp3(dim(1),dim(2)))

            do ii=1,dim(1)
                do jj=1,dim(2)
                    ! V^T
                    tmp1(ii,jj) = V(jj,ii)
                end do
            end do

            !* tmp2 = D V^T
            call dgemm('n','n',dim(1),dim(2),dim(2),1.0d0,D,dim(1),tmp1,0.0d0,tmp2)

            !* tmp3 = V tmp2 = V D V^T
            call dgemm('n','n',dim(1),dim(2),dim(2),1.0d0,V,dim(1),tmp2,0.0d0,tmp3)

            !* check that A = V D V^T
            do ii=1,dim(1)
                do jj=1,dim(2)
                    if (abs(symmetric_matrix(ii,jj)).gt.dble(1e-15)**2) then
                        atol = dble(1e-12)**2
                    end if
                    atol = abs(symmetric_matrix(ii,jj))*dble(1e-9)

                    if (abs(symmetric_matrix(ii,jj)-tmp3(ii,jj)).gt.atol) then
                        res = .false.
                    end if
                end do
            end do
            check_symmetric_eigen_decomposition = res
        end subroutine check_symmetric_eigen_decomposition
end module utility
