module features
    use config

    implicit none

    contains
        subroutine calculate_bispectrum_type1(cell,atom_positions,grid_coordinates,rcut_in,parallel)
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
            real(8),intent(in) :: grid_coordinates(:,:),rcut_in
            logical,intent(in) :: parallel

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
            rcut = rcut_in

            !* initialise shared config info
            call config_type__set_cell(cell)
            call config_type__set_local_positions(atom_positions)   

            !* cartesians of all relevant atom images 

            !* list of relevant images
            call find_neighbouring_images(neigh_images)

            !* generate cartesians of all relevant atoms
            call config_type__generate_ultracell(neigh_images)

            if (.not.parallel) then
                loop(1) = 1
                loop(2) = ngrid

                do ii=loop(1),loop(2),1
                    call config_type__generate_neighbouring_polar(grid_coordinates(:,ii),polar)

                    
                end do
            else
                write(*,*) "Not implemented"
                call exit(0)
            end if            
        end subroutine calculate_bispectrum_type1

end module features
