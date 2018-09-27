"""
module for bispectrum features around atoms
"""
from features.bispectrum_f90api import f90wrap_calculate_bispectrum_type1 as f90_bi_type1
from features.bispectrum_f90api import f90wrap_cardinality_bispectrum_type1 as f90_num_features_type1

def bispectrum(cell,atom_pos_uvw,xyz,nmax,lmax,rcut=6.0,parallel=True,form=1):
    """
    Return the bispectrum features for a seris of grid points in real space.
    These points are embedded in an infinite periodic crystal defined by
    cartesian cell vectors and fractional coordinates of atoms (single species
    only is supported)

    Arguments
    ---------
    cell, shape=(3,3)
        - cartesian coordinates of cell vectors : cell_ix is xth cartesian 
          component of ith cell vector
    
    atom_pos_uvw, shape=(N,3)
        - fractional coordinates of N atoms in local cell

    xyz, shape=(Ngrid,3)
        - cartesian coordinates of Ngrid points 

    nmax, int
        - radial term n=[1,nmax]

    lmax, int
        - spherical term l=[0,lmax]

    form, int
        - formulation of bispectrum features

    Returns
    -------
    X, shape=(Ngrid,Nfeat)

    Note
    ----
    Full periodic boundaries are implemented, periodic images of crystal 
    extend to inifinity (interactions are finite due to tapering with radius)
    """

    if form not in [1]:
        raise FeaturesError("bispectrum type {} not supported".format(form))

    num_features = getattr(f90_num_features_type1)(lmax=lmax,nmax=nmax)

    X = np.zeros((num_features,xyz.shape[0]),dtype=np.float64,order='F')

    getattr(f90_bi_type1)(cell=format_py_to_f90(cell),atom_positions=format_py_to_f90(atom_positions),\
            grid_coordinates=format_py_to_f90(xyz),rcut=rcut,parallel=parallel,lmax=lmax,nmax=nmax,X=X)

    return np.asarray(X.T,order='C')

def format_py_to_f90(array):
    """
    Python adopts C page ordering [slow,fast] indexing in contrast to fortran's
    [fast,slow] indexing. Transpose all native python arrays, check dtype and
    explicitly expect fortran page order
    """
    return np.asarray(array.T,dtype=np.float64,order='F')

class FeaturesError(Exception):
    pass
