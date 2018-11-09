#!/usr/bin/env python3

"""
boost_castep.py [-np <nproc>] [-model <loc>]

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

import sys
import os
from features.regressor import regressor

def argparse(arglist):
    """
    parse arglist into key:value pairs of arg descriptor or command line key
    and its corresponding value
    """
    def arg_error(message):
        print(__doc__)
        print("\n{}\n".format(message))
        sys.exit(0)

    def valid_model_dir(_dir,suppress_warning=False):
        # check _dir for all required model files
        dir_ok = True

        if not os.path.isdir(_dir):
            dir_ok = False
            if not suppress_warning:
                arg_error("model directory {} cannot be found".format(_dir))
       
        try:
            inst = regressor(load=_dir)
        except FileNotFoundError:
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
    
    # list of optional keys
    keys = [_c.strip("-") for _c in arglist if _c[0] == "-" or _c[:2]=="--"]

    # list of all key,value pairs with default values where applicable
    all_key_value = {"np":1,"model":{True:search_cwd_for_model(),False:None}["model" not in keys]}

    # lits of allowed optional keys
    allowed_keys = ["np","model"]

    # check if given value is valid
    value_valid = {"np":lambda x : x.isdigit() and int(x)>0,"model":valid_model_dir}

    # when values are wrong, tell user why
    value_description = {"np":"a positive integer","model":"a directory containing all necessary model files"}

    # cast value from string to chosen dtype
    value_cast = {"np":int,"model":str}

    # list of optional keys
    keys = [_c.strip("-") for _c in arglist if _c[0] == "-" or _c[:2]=="--"]
    
    # unrecognised optional keys
    unrecognised_keys = [_key for _key in keys if _key not in allowed_keys]

    if len(unrecognised_keys)!=0:
        arg_error("Unrecognised keys : {} passed to argument list".format(",".join(unrecognised_keys)))

    # check that each key has a corresponding value
    for ii,_str in enumerate(arglist):
        for _key in keys:
            if arglist[ii].strip("-")!=_key:
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

if __name__ == "__main__":
    args = argparse(sys.argv[1:])

    print(args)

















