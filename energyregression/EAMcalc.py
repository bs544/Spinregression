import numpy as np
import os
import ase
from ase.calculators.eam import EAM
from ase import Atoms

def read_header(data):
    """
    Parameters:
        data: (list) list of the lines
    Returns:
        header_data: (dict) contains information in the form of:
                    'potential_type'
                    'num_functions'
                    'num_atom_types'
    """
    start_header = None
    end_header = 0
    for i,line in enumerate(data):
        if ('#F' in line):
            start_header = i
            line = line.split()
            num_functions = int(line[-1])
        
        if ('#E' in line and start_header is not None):
            end_header = i
            break
    assert (start_header is not None), "Header reading failed"

    for i in range(start_header+1,end_header):
        line = data[i]
        if ('#T' in line):
            line = line.split()
            pot = line[-1]
        if ('#C' in line):
            line = line.split()
            num_types = len(line)-1
    
    if (num_types > 1 or pot != 'EAM'):
        raise NotImplementedError
    header_data = {'potential_type':pot,'num_functions':num_functions,'num_atom_types':num_types,'end_header_line':end_header}
    return header_data

def read_function(func,lines):
    """
    Parameters:
        func: (str) name of the function
        lines: (list) lines containing the function information
    Returns:
        func_params: (dict) each key is a parameter, eg 'cutoff' 'alpha' 'r_0'
    """
    #check that you've got the right function
    line_func = lines[0].split()[-1]
    assert(line_func==func), "Inconsistent functions: {} and {}".format(line_func,func)

    func_params = {}

    for line in lines:
        if ('rmin' not in line and 'type' not in line):
            line = line.split()
            if (len(line)>0):
                func_params[line[0]] = float(line[1]) #the final two values are max and min values, and I don't think those are necessary
    if (func=='universal'):
        #they decided to go with m and n instead of p and q like they do in the function descriptions, changing the keys to reflect this
        func_params['p'] = func_params.pop('m')
        func_params['q'] = func_params.pop('n')
    return func_params

def param_parser(name):
    """
    Parameters:
        name: (str) name of the .pot_end file
    Returns:
        params: (dict) each element is a dictionary for each function used. 
        An example would be a key like 'harmonic' with the corresponding entry being a dictionary with keys 'cutoff' 'alpha' 'r_0'
    """
    with open(name,'r') as f:
        fdata = f.readlines()
    
    data = read_header(fdata)

    functions = []
    function_starts = []
    for i in range(data['end_header_line'],len(fdata)):
        line = fdata[i]
        if ('type' in line):
            line = line.split()
            assert(len(line)==2), "Too many words in line {}".format(fdata[i])
            functions.append(line[-1])
            function_starts.append(i)
    
    assert (len(functions)==data['num_functions']), "Header data inconsistent with the rest of the file. \
        Header says {} functions when there are {} functions".format(data['num_functions'],len(functions))
    
    function_starts.append(len(fdata))
    
    function_params = {}
    for i, function in enumerate(functions):
        func_params = read_function(function,fdata[function_starts[i]:function_starts[i+1]])
        function_params[function] = func_params
    return function_params

#pairwise potentials
def harmonic(alpha,r_0,cutoff,r):
    """
    Parameters:
        alpha: (float)
        r_0: (float)
        cutoff: (float)
        r: (array), shape (N,M) interatomic distance
    Returns:
        V = alpha*(r-r_0)^2 for r< cutoff
    """
    return  np.where(r<cutoff, alpha*(r-r_0)**2, np.zeros(r.shape))

def LJ(e,s,cutoff,r):
    """
    Parameters:
        e: (float)
        s: (float)
        cutoff: (float)
        r: (array), shape(N,M) interatomic distances
    Returns:
        V = 4e*[(s/r)^12-(s/r)^6]
    """
    x = (s/r)**6
    return np.where(r<cutoff,4*e*(x**2-x),np.zeros(r.shape))

#Density functions
def csw(a_1,a_2,alpha,beta,cutoff,r):
    """
    Parameters:
        a_1: (float)
        a_2: (float)
        alpha: (float)
        beta: (float)
        cutoff: (float)
        r: (array), shape (N,M) interatomic distance
    Returns:
        rho(r) = (1+a_1*cos(alpha*r)+a_2*sin(alpha*r))/r^beta for r< cutoff
    Note:
        cos and sin functions accept inputs as radians, and I'm expecting this to be the correct form
    """
    return np.where(r<cutoff, (1+a_1*np.cos(alpha*r)+a_2*np.sin(alpha*r))/r**beta, np.zeros(r.shape))

def csw2(a_1,alpha,phi,beta,cutoff,r):
    """
    Parameters:
        a_1: (float)
        alpha: (float)
        phi: (float)
        beta: (float)
        cutoff: (float)
        r: (array), shape (N,M) interatomic distances
    Returns:
        rho(r) = (1+a_1*cos(alpha*r+phi))/r^beta
    """
    return np.where(r<cutoff, ((1+a_1*np.cos((alpha*r+phi)))/r**beta),np.zeros(r.shape))

#Embedding functions
def universal(F_0,p,q,F_1,n):
    """
    Parameters:
        F_0: (float)
        p: (float)
        q: (float)
        F_1: (float)
        n: (array), shape (N) density
    Returns:
        F(n) = F_0*[(q/(q-p))n^p - (p/(q-p))n^q] + F_1*n
    """
    return F_0*((q*(n**p) - p*(n**q))/(q-p)) + F_1*n

#Cutoff function
def f_c(r,r_c,h):
    """
    Parameters:
        r: (array), shape(N,M) distances
        r_c: (float), cutoff distance
        h: (float), parameter for cutoff
    Returns:
        f_c(r,r_c,h) = x^4/(1+x^4), where x = (r-r_c)/h
    """
    x = (r-r_c)/h
    return np.where(r<r_c,x**4/(1+x**4),np.zeros(r.shape))

#smooth cutoff for functions
def csw2_sc(a_1,alpha,phi,beta,cutoff,h,r):
    """
    See csw2 and f_c for description of parameters.
    Applies the smooth cutoff to csw2
    """
    return csw2(a_1,alpha,phi,beta,cutoff,r)*f_c(r,cutoff,h)

def LJ_sc(e,s,cutoff,h,r):
    """
    See LJ and f_c for a description of parameters
    Applies the smooth cutoff to LJ
    """
    return LJ(e,s,cutoff,r)*f_c(r,cutoff,h)

def get_total_density(r,function_params,function):
    """
    Parameters:
        r: (array), shape(N,M) interatomic distances for M atoms
        function_params: (dict) dictionary containing the parameters for the density function
        function: (str) name of the density function
    Returns:
        n: (array), shape (M) density for M atoms
    """
    if (function == 'csw'):
        a_1 = function_params['csw']['a_1']
        a_2 = function_params['csw']['a_2']
        alpha = function_params['csw']['alpha']
        beta = function_params['csw']['beta']
        cutoff = function_params['csw']['cutoff']
        densities = csw(a_1,a_2,alpha,beta,cutoff,r)
        return np.sum(densities,axis=0)
    elif (function == 'csw2'):
        a_1 = function_params[function]['a']
        alpha = function_params[function]['alpha']
        beta = function_params[function]['beta']
        phi = function_params[function]['phi']
        cutoff = function_params[function]['cutoff']
        densities = csw2(a_1,alpha,phi,beta,cutoff,r)
        return np.sum(densities,axis=0) 
    elif (function == 'csw2_sc'):
        a_1 = function_params[function]['a']
        alpha = function_params[function]['alpha']
        beta = function_params[function]['beta']
        phi = function_params[function]['phi']
        cutoff = function_params[function]['cutoff']
        h = function_params[function]['h']
        densities = csw2_sc(a_1,alpha,phi,beta,cutoff,h,r)
        # densities = np.where(densities>0,densities,np.zeros(densities.shape))
        densities = np.sum(densities,axis=0)
        return densities
    else:
        raise NotImplementedError
    return

def get_total_energy(r,function_params):
    """
    Parameters:
        r: (array), shape(N,M) interatomic distances for M atoms
        function_params: (dict) dictionary with the function names as keys, and the function parameters as the corresponding entries
    Returns:
        Energy: (array), shape (M) energies for M atoms
    """
    functions = function_params.keys()
    transfer_funcs = ['csw','csw2','csw2_sc']
    embedding_funcs = ['universal']
    pairwise_funcs = ['harmonic','lj','lj_sc']

    used_embedding_func = list(set(embedding_funcs)&set(functions))
    used_transfer_func = list(set(transfer_funcs)&set(functions))
    used_pairwise_func = list(set(pairwise_funcs)&set(functions))

    assert (len(used_embedding_func)==1), "Can only deal with one embedding function at the moment"
    assert (len(used_transfer_func)==1), "Can only deal with one transfer function at the moment"
    assert (len(used_pairwise_func)==1), "Can only deal with one pairwise function at the moment"

    if (used_pairwise_func[0] == 'harmonic'):
        func = used_pairwise_func[0]
        func_dict = function_params[func]
        pairwise_energy = harmonic(func_dict['alpha'],func_dict['r_0'],func_dict['cutoff'],r)
        pairwise_energy = 0.5*np.sum(pairwise_energy)
    elif (used_pairwise_func[0] == 'lj'):
        func = used_pairwise_func[0]
        func_dict = function_params[func]
        pairwise_energy = LJ(func_dict['epsilon'],func_dict['sigma'],func_dict['cutoff'],r)
        pairwise_energy = 0.5*np.sum(pairwise_energy)
    elif (used_pairwise_func[0] == 'lj_sc'):
        func = used_pairwise_func[0]
        func_dict = function_params[func]
        pairwise_energy = LJ_sc(func_dict['epsilon'],func_dict['sigma'],func_dict['cutoff'],func_dict['h'],r)
        pairwise_energy = 0.5*np.sum(pairwise_energy)
    else:
        raise NotImplementedError

    
    func = used_transfer_func[0]
    densities = get_total_density(r,function_params,func)
    # densities = np.square(densities)

    if (used_embedding_func[0]=='universal'):
        func = used_embedding_func[0]
        func_dict = function_params[func]
        embedding_energy = universal(func_dict['F_0'],func_dict['p'],func_dict['q'],func_dict['F_1'],densities)
        embedding_energy = np.sum(embedding_energy)
    else:
        raise NotImplementedError
    
    return pairwise_energy+ embedding_energy

def get_distances(positions,cell,cutoff):
    """
    Parameters:
        positions: (array), shape(N,3) positions of N atoms in cell (Angstroms)
        cell: (array), shape(3,3) cell vectors in Angstroms
        cutoff: (float) cutoff distance for interactions
    Returns:
        r: (array) shape (M,N) M interatomic distances for every atom in the unit cell, one for every atom in the cell and its neighbours
    """
    vect_lengths = np.linalg.norm(cell,axis=1)
    n_periodic_images = np.ceil(cutoff/vect_lengths).astype(int)
    n_periodic_images_ = 2*n_periodic_images + 1
    
    natoms = positions.shape[0]
    periodic_posns = np.zeros((np.prod(n_periodic_images_),natoms,3))

    for i in range(-n_periodic_images[0],n_periodic_images[0]+1):
        for j in range(-n_periodic_images[1],n_periodic_images[1]+1):
            for k in range(-n_periodic_images[2],n_periodic_images[2]+1):
                idx_1 = (i+n_periodic_images[0])*n_periodic_images_[1]*n_periodic_images_[2]\
                     + (j+n_periodic_images[1])*n_periodic_images_[2] + (k+n_periodic_images[2])
                periodic_posns[idx_1,:,:] = positions + i*cell[0,:] + j*cell[1,:] + k*cell[2,:]
    periodic_posns = periodic_posns.reshape(-1,3)
    displacements = np.zeros((periodic_posns.shape[0],positions.shape[0],3))
    for i in range(positions.shape[0]):
        displacements[:,i,:] = periodic_posns - positions[i,:]
    distances = np.linalg.norm(displacements,axis=2)
    keep = np.any(distances<cutoff,axis=1)
    distances = distances[keep,:]
    r = np.zeros((distances.shape[0]-1,distances.shape[1]))
    for i in range(distances.shape[1]):
        r[:,i] = distances[(distances[:,i]>0.0),i]
    return r

def get_cell_energy(cell,positions,param_dict):
    """
    Parameters:
        cell: (array) shape (3,3) cell vectors in Angstroms
        positions: (array) shape (N,3) atomic positions in Angstroms
        param_dict: (dict) dictionary of parameters for the functions
    """
    cutoff = 0
    for key in param_dict.keys():
        cutoff = max(cutoff,param_dict[key]['cutoff'])
    r = get_distances(positions,cell,cutoff)
    energy = get_total_energy(r,param_dict)
    return energy

def get_potfit_energies(cell_list,position_list,param_file):
    """
    Parameters:
        cell_list: (list) list of cell arrays
        position_list: (list) list of positions
        param_file: (str) name of file containing parameters
    Returns:
        energy_list: (list) list of cell energies
    """

    param_dict = param_parser(param_file)
    energies = []
    for i in range(len(cell_list)):
        cell = cell_list[i]
        positions = position_list[i]
        energies.append(get_cell_energy(cell,positions,param_dict))
    return energies

def get_alloy_energies(cell_list,position_list,param_file):
    """
    Parameters:
        cell_list: (list) list of cell arrays
        position_list: (list) list of positions
        param_file: (str) name of file containing parameters
    Returns:
        energy_list: (list) list of cell energies
    """
    calc = EAM(potential=param_file)

    energies = []
    for i in range(len(cell_list)):
        cell = cell_list[i]
        positions = position_list[i]
        atoms = Atoms(['Fe' for i in range(positions.shape[0])],positions=positions,cell=cell)
        atoms.set_calculator(calc)
        energy = atoms.get_potential_energy()
        energies.append(energy)
    return energies

def get_energies(cell_list,position_list,param_file):
    """
    Parameters:
        cell_list: (list) list of cell arrays
        position_list: (list) list of positions
        param_file: (str) name of file containing parameters
    Returns:
        energy_list: (list) list of cell energies
    """

    if ('.pot' in param_file):
        energies = get_potfit_energies(cell_list,position_list,param_file)
        return energies
    elif ('.alloy' in param_file):
        energies = get_alloy_energies(cell_list,position_list,param_file)
        return energies
    else:
        raise NotImplementedError


