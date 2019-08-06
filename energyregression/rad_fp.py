"""
module for 2 body features around atoms
"""
from energyregression.radial_f90api import f90wrap_calculate_local as f90_calculate_local
import numpy as np



def calculate(cell,atom_pos_uvw,nmax,rcut=6.0,parallel=False,local_form="rad_basis",weighting=None):
    """
    For each atom, calculate local radial features
    """

    if (weighting is None):
        weighting = np.ones(atom_pos_uvw.shape[0],dtype=np.float64,order='F')
    elif (not isinstance(weighting,np.ndarray)):
        assert (isinstance(weighting,list)), "Haven't implemented a conversion from type {} to array yet".format(type(weighting))
        assert (len(weighting)==atom_pos_uvw.shape[0]), "weighting is the incorrect length. Length: {}, Required Length {}".format(len(weighting),atom_pos_uvw.shape[0])
        weighting = np.array(weighting,dtype=np.float64,order='F')
    else:
        weighting = weighting.astype(np.float64).order('F')
    
    assert (local_form is not None), "No fingerprint type selected"

    if local_form is not None:
        X = local_features(cell=cell,atom_pos_uvw=atom_pos_uvw,nmax=nmax,rcut=rcut,\
            weighting=weighting,parallel=parallel,form=local_form)

    return X

def local_features(cell,atom_pos_uvw,nmax,weighting,rcut=6.0,parallel=True,form="rad_basis",buffersize=1000):
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

    if form not in ["rad_basis","2bodyBP"]:
        raise FeaturesError("radial fingerprint type {} not supported".format(form))

    c_type = {"rad_basis":0,"2bodyBP":1}[form]
    if (c_type == 1): raise NotImplementedError
    num_features = nmax

    X = np.zeros((num_features,atom_pos_uvw.shape[0]),dtype=np.float64,order='F')

    f90_calculate_local(cell=format_py_to_f90(cell),atom_positions=format_py_to_f90(atom_pos_uvw),\
            weightings=weighting,rcut=rcut,parallel=parallel,nmax=nmax,calc_type=c_type,\
            buffer_size=int(buffersize),x=X)

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
