#!/bin/bash

modulename="radial_f90api"

f1="io"
f2="config"
f3="utility"
f4="boundaries"
f5="features"

FFLAGS='-W -Wall -pedantic -fPIC -llapack -lblas -fopenmp -O2'

for file in {$f1,$f2,$f3,$f4,$f5}
do
    for suffix in {"o","mod"}
    do
        rm $file"."$suffix
    done
    rm "f90wrap_"$file".f90"

    gfortran -c $file".f90" $FFLAGS
done

functions_to_expose="calculate_local"
modules_to_expose="features.f90 utility.f90"

# remove this line on Cottrell
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgfortran.so.3

f90wrap -m $modulename $modules_to_expose -k kind_map -S 12 --only $functions_to_expose
f2py -c -m $modulename -llapack -lblas f90wrap_*.f90 *.o --f90flags="-fPIC -fopenmp -llapack -lblas -O2" -lgomp --fcompiler=gfortran

#rm $modulename".py"
