# Uncertainty Quantification in density regression

This Python module uses f90wrap and f2py to generate an interface to bispectrum 
calculations performed in fortran (5.x>0). Recommended OS is Linux, no other
platforms have been tested. Place or soft link the features directory within the 
site-packages directory on a python3 distribution. Before attempting to import,
run build.sh within the features directory to compile the fortran. 

The data_sets directory contains CASTEP input files that generate a series of
small data sets for density regression.
