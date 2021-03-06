"""
module for bispectrum features around atoms
"""
from features.powerspectrum_f90api import f90wrap_calculate_local as f90_calculate_local
from features.powerspectrum_f90api import f90wrap_calculate_global as f90_calculate_global
from features.powerspectrum_f90api import f90wrap_check_cardinality as f90_num_features
import numpy as np

def fp_length(nmax,lmax,form):
    # just a wrapper for f90_num_features
    if form not in ["powerspectrum","bispectrum","axial-bispectrum"]:
        raise FeaturesError("bispectrum type {} not supported".format(form))

    c_type = {"powerspectrum":0,"bispectrum":1,"axial-bispectrum":2}[form]
    num_features = f90_num_features(lmax=lmax,nmax=nmax,calc_type=c_type)
    return num_features

def calculate(cell,atom_pos_uvw,xyz,nmax,lmax,rcut=6.0,parallel=True,local_form="powerspectrum",global_form="powerspectrum",gnmax=None,glmax=None,grcut=None,weighting=None):
    """
    For each density grid point in xyz, calculate local power/bi-spectrum
    features. Compute the global crystal representation using power/bi-spectrum
    features once and concacentate this to all grid points
    """

    if (weighting is None):
        weighting = np.ones(atom_pos_uvw.shape[0],dtype=np.float64,order='F')
    elif (not isinstance(weighting,np.ndarray)):
        assert (isinstance(weighting,list)), "Haven't implemented a conversion from type {} to array yet".format(type(weighting))
        assert (len(weighting)==atom_pos_uvw.shape[0]), "weighting is the incorrect length. Length: {}, Required Length {}".format(len(weighting),atom_pos_uvw.shape[0])
        weighting = np.array(weighting,dtype=np.float64,order='F')
    else:
        weighting = weighting.astype(np.float64).order('F')
    
    assert (local_form is not None or global_form is not None), "No fingerprint type selected"

    if local_form is not None:
        localX = local_features(cell=cell,atom_pos_uvw=atom_pos_uvw,xyz=xyz,nmax=nmax,lmax=lmax,rcut=rcut,\
            weighting=weighting,parallel=parallel,form=local_form)

    if global_form is not None:
        gnmax = nmax if (gnmax is None) else gnmax
        glmax = lmax if (glmax is None) else glmax
        grcut = 6.0 if (grcut is None) else grcut
        globalX = global_features(cell=cell,atom_pos_uvw=atom_pos_uvw,nmax=gnmax,lmax=glmax,weighting=weighting,rcut=grcut,form=global_form)
        # X = np.hstack(( localX , np.tile(globalX,(localX.shape[0],1)) ))
    if global_form is None:
        X = localX
    elif local_form is None:
        X = np.tile(globalX,(xyz.shape[0],1))
    else:
        X = np.hstack(( localX , np.tile(globalX,(localX.shape[0],1)) ))

    return X

def local_features(cell,atom_pos_uvw,xyz,nmax,lmax,weighting,rcut=6.0,parallel=True,form="powerspectrum",buffersize=1000):
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

    if form not in ["powerspectrum","bispectrum","axial-bispectrum"]:
        raise FeaturesError("bispectrum type {} not supported".format(form))

    c_type = {"powerspectrum":0,"bispectrum":1,"axial-bispectrum":2}[form]
    num_features = f90_num_features(lmax=lmax,nmax=nmax,calc_type=c_type)

    X = np.zeros((num_features,xyz.shape[0]),dtype=np.float64,order='F')

    f90_calculate_local(cell=format_py_to_f90(cell),atom_positions=format_py_to_f90(atom_pos_uvw),\
            grid_coordinates=format_py_to_f90(xyz),weightings=weighting,rcut=rcut,parallel=parallel,lmax=lmax,nmax=nmax,calc_type=c_type,\
            buffer_size=int(buffersize),x=X)

    return np.asarray(X.T,order='C')


def global_features(cell,atom_pos_uvw,nmax,lmax,weighting,rcut=6.0,form="powerspectrum",buffersize=1000):
    """
    Return the global power/bi-spectrum vector for a crystal using tapered
    local approximations for the radial bases
    """
    if form not in ["powerspectrum","bispectrum","axial-bispectrum"]:
        raise FeaturesError("bispectrum type {} not supported".format(form))

    c_type = {"powerspectrum":0,"bispectrum":1,"axial-bispectrum":2}[form]
    num_features = f90_num_features(lmax=lmax,nmax=nmax,calc_type=c_type)

    X = np.zeros(num_features,dtype=np.float64,order='F')

    f90_calculate_global(cell=format_py_to_f90(cell),atom_positions=format_py_to_f90(atom_pos_uvw),\
            weightings=weighting,rcut=rcut,lmax=lmax,nmax=nmax,calc_type=c_type,buffer_size=int(buffersize),x=X)

    return np.asarray(X,order='F')

def format_py_to_f90(array):
    """
    Python adopts C page ordering [slow,fast] indexing in contrast to fortran's
    [fast,slow] indexing. Transpose all native python arrays, check dtype and
    explicitly expect fortran page order
    """
    return np.asarray(array.T,dtype=np.float64,order='F')

class FeaturesError(Exception):
    pass
