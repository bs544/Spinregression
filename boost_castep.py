#!/usr/bin/env python3

"""
boost_castep.py <sysname> [-np <nproc>] [-model <loc>]

===========================
Mandatory arguments <value>
===========================

value           description
---------------------------------------------------------------------------
sysname         run the KS DFT calculation with inputs sysname.(cell/param)


======================================
Optional arguments : [-(-)key <value>]
======================================

key     value   description
----------------------------------------------------------------------------
np      nproc   creates an instance of mpi castep with nproc processors when
                nproc>1, otherwise when nproc=1, serial castep is run. By
                default, np = 1.

model   loc     the relative or absolute location on disk of the directory
                containing all model files for density regression you want to 
                use. By default, script searches working directory for 
                appropriate looking directory
"""

import time
t0 = time.time()
import sys
import os
import pickle
from features.predictor import prediction_MLPRegressor as predictor
from features.pca import generate_data
from features.util import tapering,write_density_to_disk
import numpy as np
import itertools

def argparse(arglist):
    """
    parse arglist into key:value pairs of arg descriptor or command line key
    and its corresponding value
    """
    def arg_error(message):
        print(__doc__)
        print("\nUser Message:\n{}\n".format(message))
        sys.exit(0)

    def valid_model_dir(_dir,suppress_warning=False):
        # check _dir for all required model files
        dir_ok = True

        if not os.path.isdir(_dir):
            dir_ok = False
            if not suppress_warning:
                arg_error("model directory {} cannot be found".format(_dir))
    
        if len([_f for _f in os.listdir(_dir) if "netparams" in _f and ".pckl" in _f])==1:
            with open("{}/{}".format(_dir,[_f for _f in os.listdir(_dir) if "netparams" in _f and ".pckl" in _f][0]),"rb") as f:
                data = pickle.load(f)
                if not all([_key in data.keys() for _key in ["preconditioning","activation","hidden_layer_sizes","layer_units",\
                        "weights","biases","Nensemble"]]):
                    if not suppress_warning: arg_error("netparams file in directory {} missing required data".format(_dir))
                    dir_ok = False
        else:
            if not suppress_warning: arg_error("model directory {} does not contain a netparams file".format(_dir))
            dir_ok = False
        if len([_f for _f in os.listdir(_dir) if "pca" in _f and ".pckl" in _f])!=1:
            # need pca file as contains info about features
            dir_ok = False

        return dir_ok

    def search_cwd_for_model():
        # if this is called, no optional arg is given, all up to this routine
        valid_model_dirs = []
        for _dir in [_f for _f in os.listdir('.') if os.path.isdir(_f)]:
            if valid_model_dir(_dir,suppress_warning=True):
                valid_model_dirs.append(_dir)
        if len(valid_model_dirs)==0:
            arg_error("no model directory found in current directory, need to specify directory location")
        elif len(valid_model_dirs)>1:
            arg_error("more than 1 appropriate model directories found in current directory, need to specify which to use")
        return valid_model_dirs[0]               
    
    def check_cellparam(sysname):
        return all(["{}.{}".format(sysname,_suffix) in os.listdir('.') for _suffix in ["cell","param"]])

    # list of optional keys
    keys = [_c.strip("-") for _c in arglist if _c[0] == "-" or _c[:2]=="--"]

    # list of all key,value pairs with default values where applicable
    all_key_value = {"np":1,"model":{True:search_cwd_for_model(),False:None}["model" not in keys],"sysname":None}

    # lits of allowed optional keys
    allowed_keys = ["np","model"]

    # check if given value is valid
    value_valid = {"np":lambda x : x.isdigit() and int(x)>0,"model":valid_model_dir,"sysname":check_cellparam}

    # when values are wrong, tell user why
    value_description = {"np":"a positive integer","model":"a directory containing all necessary model files",\
            "sysname":"be the suffix to cell and param files in the current directory"}

    # cast value from string to chosen dtype
    value_cast = {"np":int,"model":str,"sysname":str}

    # list of optional keys
    keys = [_c.strip("-") for _c in arglist if _c[0] == "-" or _c[:2]=="--"]
    
    # unrecognised optional keys
    unrecognised_keys = [_key for _key in keys if _key not in allowed_keys]

    if len(unrecognised_keys)!=0:
        arg_error("Unrecognised keys : {} passed to argument list".format(",".join(unrecognised_keys)))

    # mandatory arg positions
    arg_indices = {"sysname":0}

    if len(arglist)<len(arg_indices):
        arg_error("Invalid number of arguments. {} given, require {}".format(len(arglist),len(arg_indices)))
    else:
        # parse mandatory args
        for _arg in arg_indices.keys():
            all_key_value[_arg] = value_cast[_arg](arglist[arg_indices[_arg]])

            if not value_valid[_arg](all_key_value[_arg]):
                arg_error("value for key {} must {}".format(_arg,value_description[_arg]))

    # check that each key has a corresponding value
    for ii,_str in enumerate(arglist):
        for _key in keys:
            if arglist[ii].strip("-")!=_key or arglist[ii][0]!="-":
                continue

            value_missing = False
            try:
                if arglist[ii+1][0] == "-":
                    value_missing = True
            except IndexError: 
                value_missing = True

            if value_missing:
                arg_error("value for key {} is missing from argument list".format(_key))
            else:
                # check that type and bounds of given value are OK
                if not value_valid[_key](arglist[ii+1]):
                    arg_error("value for key {} must be {}".format(_key,value_description[_key]))
                else:
                    # value OK, add to list
                    all_key_value[_key] = value_cast[_key](arglist[ii+1]) 
    
    # if model is not specified in arglist, check current directory for 
    # valid model directories to use, only use if exactly 1 choice is found
    
    return all_key_value

class toy_gip():
    def __init__(self):
        self.supercells = None

def get_crystal(sysname):
    with open("{}.cell".format(sysname)) as f:
        # may cause problems with element type
        flines = [_l.lower() for _l in f.readlines()]
    
        indices = {"lattice_cart":[],"positions_frac":[]}
        for _key in indices.keys():
            indices[_key] = [ii for ii,_l in enumerate(flines) if "block {}".format(_key) in _l]  
            if len(indices[_key])!=2:
                print("only lattice_cart and positions_frac supported for cell vectors and atom positions currently")
                sys.exit()
        
        prop = {"lattice_cart":None,"positions_frac":None}
        for _attr in prop:
            prop[_attr] = [[_v for _v in flines[_idx].split()] for _idx in range(indices[_attr][0]+1,indices[_attr][1])] 

        # importing parsers has loads of overhead, remove by using toy class instance
        gip = toy_gip
        gip.supercells = [{"cell":np.asarray(prop["lattice_cart"],dtype=np.float64),\
                "species":[_x[0] for _x in prop["positions_frac"]],\
                "positions":np.asarray([_x[1:] for _x in prop["positions_frac"]],dtype=np.float64),\
                "edensity":{"xyz":None,"density":None}}]
    f.close()

    try:
        # get fine fft grid dimension
        os.system("{} -g {}".format(os.environ["CASTEP_SERIAL_BOOST"],sysname))
    except KeyError:
        print("Need to specify export CASTEP_SERIAL_BOOST=/path/to/modified/castep/serial")
        sys.exit(0)

    if not os.path.exists("{}.grid".format(sysname)):
        print("Error generating fine fft grid dimensions")
        sys.exit(0)
   
    fft_grid_dim = np.loadtxt("{}.grid".format(sysname),dtype=int)

    def product(shape, axes):
        prod_trans = list(zip(*itertools.product(*(range(shape[axis]) for axis in axes))))

        prod_trans_ordered = [None] * len(axes)
        for i, axis in enumerate(axes):
            prod_trans_ordered[axis] = prod_trans[i]
        return [_a for _a in zip(*prod_trans_ordered)]

    # generate grid for density field 
    fft_grid = np.asarray(product(fft_grid_dim,(2,1,0)) , dtype=np.float64)
    
    # convert grid coordinates to fractional (divide by number points per dimension)
    norm_const = np.tile(1.0/fft_grid_dim,(fft_grid.shape[0],1))
       
    # cartesian coordinates of grid points       
    gip.supercells[0]["edensity"]["xyz"] = np.dot(fft_grid*norm_const , gip.supercells[0]["cell"])
        
    # tidy up
    os.remove("{}.castep".format(sysname))
    os.remove("{}.0001.err".format(sysname))
    os.remove("{}.grid".format(sysname))

    return gip,fft_grid_dim

def fetch_transformation_for_net(model):
    if not os.path.exists("{}/{}.pckl".format(model,model)):
        print("Cannot find regressor model file {}/{}.pckl".format(model,model))
        sys.exit(0)
    
    with open("{}/{}.pckl".format(model,model),"rb") as f:
        data = pickle.load(f)
    f.close()

    return data["train_data"]

def fetch_tapering_function(model):
    if os.path.exists("{}/tapering-{}.pckl".format(model,model)):
        # tapering active
        with open("{}/tapering-{}.pckl".format(model,model),"rb") as f:
            data = pickle.load(f)
    
        return lambda x: tapering(x,data["xcut"],data["scale"])
    else:
        return lambda x : 1.0
    

def generate_initial_density(args):
    # raw features and pca reduction
    feats = generate_data(load="{}".format(args["model"]))
    
    # train input to nets scaled so that <X>=0 and var(X)=1 along each dimension
    net_input_transform = fetch_transformation_for_net(args["model"])

    # ensemble of fully connected feed forward nets
    network_ensemble = predictor("{}".format(args["model"]))

    # tapering function for expected log(variance)
    tapering_func = fetch_tapering_function(args["model"])

    # read .cell for cell vectors, grid points, atom positions
    gip,fft_grid = get_crystal(args["sysname"])

    t1 = time.time()

    # pca reduced input
    X,_ = feats.predict(gip)

    t2 = time.time()

    # need to rescale reduced pca input such that <X>=0, var(X)=1 along each axis
    X = net_input_transform.get_xs_standardized(X)

    t3 = time.time()

    ymean,ystd = network_ensemble.predict(X)

    t4 = time.time()

    # apply tapering, identity when no tapering pckl file is present
    taper = tapering_func(np.average(np.log(np.square(ystd))))

    gip.supercells[0]["edensity"]["density"] = ymean*taper

    # volume of primitive cell in real space
    cell_volume = abs(np.dot(gip.supercells[0]["cell"][0],np.cross(gip.supercells[0]["cell"][1],gip.supercells[0]["cell"][2])))

    # write unformatted density file, castep takes units=[e] rather than untis=[e/A^3]
    write_density_to_disk(gip.supercells[0]["edensity"]["density"]*cell_volume,\
            fft_grid,"{}.initial_den".format(args["sysname"]))

def run_castep(args):
    if args["np"]>1:
        print("only serial computation currently supported")
        sys.exit(0)

    os.system("{} {}".format(os.environ["CASTEP_SERIAL_BOOST"],args["sysname"]))

    # tidy up
    #os.remove("{}.initial_den".format(args["sysname"]))

if __name__ == "__main__":
    args = argparse(sys.argv[1:])

    generate_initial_density(args)
    
    run_castep(args)

    print(time.time()-t0)














