import numpy as np
from parsers.dft.parser_castep import parse
#parser code courtesy of Andrew Fowler
from parsers.structure_class import supercell
import pickle
import os
import copy

#I'm going to remove the threshold stuff, since it's more transparent if that's done outside the class
#I'm also going to clean up the code a bit
#I've changed the spin condition to be when the number of spins >0, this was just so it would be easier to deal with non-collinear spins if and when the time comes.

class castep_data():
    def __init__(self,dataPath="./",filename="castep_density_data",savedir="./Pickle_data/",datalist=None,nspins=0,verbose=False):
        """
        This class reads in data of the form of a *.castep *.den_fmt *_initial.den_fmt set of files for each cell 
        and outputs a list of dictionaries, one for each cell in self.data, each dictionaly contains:
            ['cell']: cell vectors
            ['elements']: atomic species identifiers
            ['positions']: cartesian (or is it fractional?) atomic coordinates
            ['xyz']: cartesian coordinates of each point of the grid on which density is specified
            ['edensity']: total electron density (spin up + spin down) on a grid
            ['init_edensity']: initial electron density
        
        If self.nspins>0 then it also contains:
            ['sdensity']: spin density (spin up - spin down) on a grid.
            ['init_sdensity']: initial spin density
        
        The data is taken from the directory specified in self.path, and each dictionary is saved in self.savedir
        with the filename self.filename and numbers appended to the end

        If verbose is on, then you get a commentary of what's happening
        """
        self.path = dataPath
        self.data = []
        self.relevantFileTypes = ['castep','den_fmt']
        self.filename = filename
        self.savedir = savedir
        self.nspins = nspins
        self.datalist = datalist
        self.verbose = verbose
        self.include_init = None

        self.keys = ["edensity","xyz","cell","elements","positions","forces"]

        self.init_keys = ['init_edensity']

        if self.nspins>0:
            self.keys.append('sdensity')
            self.init_keys.append('init_sdensity')

    def get_densities(self,name):
        """
        Takes a filename and parses the .den_fmt file using Andrew Fowler's castep parsing module
        Returns a dictionary containing:
            ['xyz']: cartesian coordinates of the grid points on which the density is evaluated (Angstroms)
            ['edensity']: total electron density (spin up + spin down) on a grid
            ['init_edensity']: initial electron density
        """
        #name is the prefix for the .den_fmt file
        #returns a dictionary containing densities and positions and cell parameters taken from the .den_fmt file given
        dendict = {}
        filename = '{}{}'.format(self.path,name)
        type = 'den_fmt'
        l = len(type)
        if (type not in name[-l:]):
            print('need to pass a .den_fmt file to get_densities()')
            return
        den_parser = parse(filename,type)
        den_parser.run()
        den_cell = den_parser.get_supercells()[0]
        if (self.nspins>0):
            den = den_cell.get_sdensity()
        else:
            den = den_cell.get_edensity()

        dendict['xyz'] = den['xyz']
        if (self.nspins>0):
            dendict['edensity'] = den['edensity']
            dendict['sdensity'] = den['sdensity']
        else:
            dendict['edensity'] = den['density']


        return dendict

    def get_atoms(self,name):
        """
        Takes a filename and parses the .castep file using Andrew Fowler's castep parsing module
        Returns a dictionary containing:
            ['cell']: a 3x3 matrix of cartesian cell vectors (angstroms)
            ['elements']: indicators for the atomic species
            ['positions']: cartesian coordinates of atoms (angstroms)
            ['forces']: force vectors for each atom (eV/angstrom)
        """
        
        ftype = 'castep'
        l = len(ftype)
        filename = '{}{}'.format(self.path,name)

        assert ftype in name[-l:], 'need to pass a .castep file to the get_positions() function of the parser'

        celldict = {}

        pos_parser = parse(filename,ftype)
        pos_parser.run()
        pos_cell = pos_parser.get_supercells()[0]
        posns = pos_cell.get_positions()
        species = pos_cell.get_species()
        cell = pos_cell.get_cell()
        forces = pos_cell.get_forces()

        #positions are relative coordinates when they come out
        #multiply by cell values to get coordinates in Angstroms
        for i in range(posns.shape[0]):
            posns[i,:] = np.dot(cell,posns[i,:])

        celldict['cell'] = cell
        celldict['elements'] = species
        celldict['positions'] = posns
        celldict['forces'] = forces

        return celldict

    def get_diff(self,init_dict,fin_dict):
        diff_dict = {}
        #check the positions match
        if(np.array_equal(fin_dict['xyz'],init_dict['xyz'])):
            diff_dict['xyz'] = init_dict['xyz']
            diff_dict["fin_density"] = fin_dict["density"]
            diff_dict["fin_sdensity"] = fin_dict["sdensity"]
            diff_dict['density'] = fin_dict['density']-init_dict['density']
            if (self.nspins>0):
                diff_dict['sdensity'] = fin_dict['sdensity']-init_dict['sdensity']
        else:
            print('Incompatible inital and final density dictionaries')
        return diff_dict

    def combine_den_dicts(self,init_dict,fin_dict):
        """
        Combines the two dictionaries while checking that the grids are compatible
        Returns a combined grid that has the following keys:
            ['xyz'], ['edensity'], ['init_edensity'] 
        As well as: 
            ['sdensity'], ['init_sdensity']
        if self.nspins>0
        """

        combo_dict = {}

        try:
            assert np.array_equal(init_dict['xyz'],fin_dict['xyz']), "Initial and final density grids are incompatible"
        except AssertionError as error:
            if (self.verbose):
                print(error)
            Good_combo = False
            return combo_dict, Good_combo
        
        combo_dict['xyz'] = init_dict['xyz']
        combo_dict['init_edensity'] = init_dict['edensity']
        combo_dict['edensity'] = fin_dict['edensity']

        if (self.nspins>0):
            combo_dict['sdensity'] = fin_dict['sdensity']
            combo_dict['init_sdensity'] = init_dict['sdensity']

        Good_combo = True

        return combo_dict, Good_combo  

    def check_duplicated_dicts(self,init_dendict,fin_dendict):
        """
        Checks whether the densities in the initial and final dictionaries are the same
        If this is the case then there was likely an issue with the data collection and the function returns true
        """      
        duplicated = False

        init_eden = init_dendict['edensity']
        fin_eden = fin_dendict['edensity']

        ediff = fin_eden - init_eden
        if (sum(ediff)==0):
            duplicated = True
        
        if (self.nspins>0):
            init_sden = init_dendict['sdensity']
            fin_sden = fin_dendict['sdensity']
            sdiff = fin_sden - fin_eden
            if (sum(sdiff)==0):
                duplicated = True
        
        return duplicated

    def loadFromPickle(self):
        """
        Loads the castep data from self.datadir. If self.datalist is not None, 
        then it will try to get that specific data out
        """
        saved_data=[]

        #Check that the save directory exists and contains files
        try:
            assert os.path.isdir(self.savedir), "The specified directory {} doesn't exist".format(self.savedir)
        except AssertionError as error:
            if (self.verbose):
                print(error)
            Good_load=False
            return saved_data, Good_load
        
        datafiles = sorted(os.listdir(self.savedir))

        try:
            assert datafiles is not None, "The directory {} is empty".format(self.savedir)
        except AssertionError as error:
            if (self.verbose):
                print(error)
            Good_load=False
            return saved_data, Good_load
        
        #Now to get the data
        if (self.verbose):
            print('Specified directory {} exists and is not empty, proceeding to extract data'.format(self.savedir))
        

        for i , fname in enumerate(datafiles):
            if (self.datalist is None or i in self.datalist):
                with open('{}{}'.format(self.savedir,fname),'rb') as f:
                    saved_dict = pickle.load(f)
                
                #use_keys are the likely keys used by the saved dictionary                
                use_keys = copy.copy(self.keys)
                
                #check if the saved dictionary includes initial values, and if so, add them to used_keys
                if (any(key in saved_dict.keys() for key in self.init_keys)):
                    use_keys.extend(self.init_keys)
                
                try:
                    assert all(key in saved_dict.keys() for key in use_keys) and \
                        all(key in use_keys for key in saved_dict.keys()), "Saved dictionary has an incompatible set of keys"
                except AssertionError as error:
                    if (self.verbose):
                        print(error)
                        print('Skipping file {}'.format(fname))
                    continue
                
                saved_data.append(saved_dict)

        #Check how many of the files were successfully loaded. If the answer is none, then the load was bad.
        if (len(saved_data)==len(datafiles) and len(datafiles)>0):
            if (self.verbose):
                print('Good load, no skips')
            Good_load = True
            for data in saved_data:
                self.include_init = True
                if ('init_edensity' not in data.keys() and 'init_sdensity' not in data.keys()):
                    self.include_init = False
        elif (len(saved_data)>0):
            if (self.verbose):
                nskips = len(datafiles)-len(saved_data)
                print('{} files skipped out of {}'.format(nskips,len(saved_data)))
            Good_load = True
            #Check if initial densities are included:
            self.include_init = True
            for data in saved_data:
                if ('init_edensity' not in data.keys() and 'init_sdensity' not in data.keys()):
                    self.include_init = False
        else:
            if (self.verbose):
                print('No files loaded from {}'.format(self.savedir))
            Good_load = False

        return saved_data, Good_load

    def get_names(self):
        """
        Gets the names of the *.castep *.den_fmt and possibly *_initial.den_fmt files
        Checks that each * has at least a .castep and .den_fmt file, and then adds them to the names
        If all of the names also have an associated *_initial.den_fmt file, then self.include_init is set to True
        Otherwise it is set to false.
        """
        names = []
        self.include_init = True

        #Check that the data directory exists and contains files
        try:
            assert os.path.isdir(self.path), "The specified directory {} doesn't exist".format(self.path)
        except AssertionError as error:
            if (self.verbose):
                print(error)
            Good_load=False
            return names, Good_load
        
        files = sorted(os.listdir(self.path))

        try:
            assert files is not None, "The directory {} is empty".format(self.path)
        except AssertionError as error:
            if (self.verbose):
                print(error)
            Good_load=False
            return names, Good_load
        
        #Now to get the names
        filenames = {'casnames':[],'init_dennames':[],'fin_dennames':[]}
        for f in files:
            try:
                fname,ftype = f.split('.')
            except:
                if (self.verbose):
                    print("{} is an invalid file type, skipping".format(f))
                continue
            
            try:
                assert (ftype in self.relevantFileTypes), "{} has type {}, which is not allowed, skipping file".format(fname,ftype)
            except AssertionError as error:
                if (self.verbose):
                    print(error)
                continue
            
            if (ftype == 'castep'):
                filenames['casnames'].append(fname)
            elif (ftype == 'den_fmt'):
                if ('_initial' in fname):
                    filenames['init_dennames'].append(fname[:-8])
                else:
                    filenames['fin_dennames'].append(fname)
        
        for name in filenames['casnames']:
            if (name in filenames['fin_dennames']):
                names.append(name)
                if (name not in filenames['init_dennames'] ):
                    self.include_init = False
            elif(self.verbose):
                print("{}.castep doesn't have a complete set of files, rejecting this file")
        
        if (len(names)>0):
            Good_read = True
        else:
            Good_read = False
            if (self.verbose):
                print("Didn't successfully get any names")
        return names, Good_read

    def save(self,save_dict,filename):
        """
        Saves an individual dictionary, save_dict, in self.savedir
        If self.savedir doesn't already exist, it creates the directory
        """

        try:
            assert os.path.isdir(self.savedir), "{} doesn't yet exist, creating directory".format(self.savedir)
        except AssertionError as error:
            if (self.verbose):
                print(error)
            os.mkdir(self.savedir)
        
        if (self.verbose):
            print("Saving castep data into {}{}.pckl".format(self.savedir,filename))
        
        with open('{}{}.pckl'.format(self.savedir,filename),'wb') as f:
            pickle.dump(save_dict,f)
        return

    def load_castepData(self,load=True,save=True):
        """
        Returns list of dictionaries. Each dictionary contains the data of a single file
        
        If load is true, will attempt to load the data from .pckl files in self.savedir
        
        If load isn't true, or the load fails, it parses the files in self.path instead
            If *_initial.den_fmt files are included in the directory, self.load_init will be True
            and these files will be parsed as well. Otherwise, just the final information is parsed.

        If save is true, the data will be saved in .pckl files in self.savedir
        """

        if (load):
            self.data, load_status = self.loadFromPickle()
            if (load_status):
                return
            elif (self.verbose):
                print('Load from Pickle files failed, loading from castep data in {}'.format(self.path))

        names, read_status = self.get_names()
        if (not read_status):
            if (self.verbose):
                print('Aborting data gathering')
            return
        

        for i,name in enumerate(names):
            if (self.datalist is None or i in self.datalist):
                celldict = self.get_atoms('{}{}'.format(name,'.castep'))
                fin_dendict = self.get_densities('{}{}'.format(name,'.den_fmt'))
                if (self.include_init):
                    init_dendict = self.get_densities('{}{}{}'.format(name,'_initial','.den_fmt'))
                    duplicated = self.check_duplicated_dicts(init_dendict,fin_dendict)
                    if (duplicated):
                        if (self.verbose):
                            print("{} has identical initial and final densities, removing from dataset".format(name))
                        continue
                    dendict,combo_status = self.combine_den_dicts(init_dendict,fin_dendict)

                    if (not combo_status):
                        if (self.verbose):
                            print('Combination of density dictionaries failed, aborting')
                        return

                else:
                    dendict = fin_dendict
                celldict.update(dendict)


                if (save):
                    self.save(celldict,'{}{}'.format(self.filename,i))
                self.data.append(celldict)
        return
    
    def get_all_data(self,cells=None,fpargs=None,sampargs=None):
        """
        Parameters:
            cells: (int) number of cells to get the data for
                   (list) specific cells to get the data for
                   If None, then all of the cells available are processed
            fpargs: (dict) arguments for fingerprint calculations. Expected to contain some of:
                        local_fptype, global_fptype, nmax, lmax, rcut, gnmax, glmax, grcut (g for global, if that wasn't clear)
                    If None, then default values are used
            sampargs: (dict) arguments for sampling from data. Expected to contain some of:
                    sampling_type and depending on sampling type: weighting, high_fraction, low_fraction, threshold, cutoff
                    If None, then no sampling is done

        This assumes that you have already loaded the data and that it is stored in self.data
        A single dictionary is returned with all of the data within it
        The returned dictionary contains:
            ['xyz']
            ['edensity']
            ['fp']

            if _initial.den_fmt files are included, as indicated by self.include_init, it also contains:
                ['init_edensity']
            
        and if self.nspins>0:
            ['sdensity']

            if _initial.den_fmt files are included, it also contains:
                ['init_sdensity']

        where ['fp'] is the fingerprint of atomic environments
        fptype, nmax, lmax and rcut are all values for the fingerprint
        
        """

        #fingerprints section almost exclusively the work of Andrew Fowler, only exception is the axial bispectrum
        from features.bispectrum import calculate
        from features.sample_idx import get_points_near_atoms, get_sampled_idx
        import time

        data = {}

        default_fpargs = {'nmax':4,'lmax':4,'local_fptype':'axial-bispectrum','global_fptype':'powerspectrum','gnmax':None,'glmax':None,'grcut':None}

        allowed_sampling_methods = ['value_weighted','split','proximity']
        default_sampargs = {'value_weighted':{'weighting':'linear'},\
            'split':{'high_fraction':0.95,'low_fraction':None,'train_ratio':0.75,'threshold':None},\
            'proximity':{'cutoff':0.5}}
        if (fpargs is None):
            fpargs = default_fpargs
        else:
            for key,value in enumerate(default_fpargs):
                if (value not in fpargs.keys()):
                    fpargs[value] = default_fpargs[value]
        
        if (sampargs is not None):
            if (sampargs['method']=='uniform'):
                if (self.verbose):
                    print("Redundant method, no sampling done")
                sampargs = None
            elif (sampargs['method'] not in allowed_sampling_methods):
                if (self.verbose):
                    print("{} not implemented, skipping sampling".format(sampargs['method']))
                sampargs = None
            else:
                sampling_dict = default_sampargs[sampargs['method']]
                for key in sampling_dict.keys():
                    if key in sampargs.keys():
                        sampling_dict[key] = sampargs[key]
                


        if (cells is None):
            l = len(self.data)
            cells = list(np.linspace(0,l-1,l).astype(int))
        elif (isinstance(cells,int)):
            cells = list(np.linspace(0,cells-1,cells).astype(int))
        try:
            assert isinstance(cells,list), "Invalid data type for cells, aborting"
        except AssertionError as error:
            if (self.verbose):
                print(error)
            good_data = False
            return data, good_data
        
        xyz = []
        fp = []
        edensity = []
        init_edensity = []
        
        if (self.nspins>0):
            sdensity = []
            init_sdensity = []
        
        start = time.time()
                
        for i in cells:

            cell = self.data[i]['cell']
            positions = self.data[i]['positions']

            xyz_ = self.data[i]['xyz']
            edensity.append(self.data[i]['edensity'])

            if (sampargs is not None):
                if (sampargs['method'] == 'proximity'):
                    samp_idx = get_points_near_atoms(xyz_,positions,cell,sampling_dict['cutoff'])
                elif (sampargs['method'] in ['value_weighted','split']):
                    samp_idx = get_sampled_idx(edensity,sampargs['method'],sampling_dict)
            else:
                samp_idx = [i for i in range(len(xyz_))]

            xyz_ = xyz_[samp_idx,:]
            edensity[i] = edensity[i][samp_idx]

            xyz.append(xyz_)
            
            if (self.include_init):
                init_edensity.append(self.data[i]['init_edensity'][samp_idx])
            
            fp_ = calculate(cell,positions,xyz_,fpargs['nmax'],fpargs['lmax'],rcut=fpargs['rcut'],local_form=fpargs['local_fptype'],global_form=fpargs['global_fptype'])
            fp.append(fp_)

            if (self.nspins>0):
                sdensity.append(self.data[i]['sdensity'][samp_idx])
                if (self.include_init):
                    init_sdensity.append(self.data[i]['init_sdensity'][samp_idx])

        
        if (self.verbose):
            print("Data collected, time taken {0:.2f}s ".format(time.time()-start))
        
        fp = np.concatenate(fp,axis=0)
        xyz = np.concatenate(xyz,axis=0)
        edensity = np.concatenate(edensity,axis=0)
        if (self.include_init):
            init_edensity = np.asarray(init_edensity)

        if (self.nspins>0):
            sdensity = np.concatenate(sdensity,axis=0)
            if (self.include_init):
                init_sdensity = np.concatenate(init_sdensity,axis=0)

        #num_gridpoints = fp.shape[0]*fp.shape[1]
        num_gridpoints = fp.shape[0]
        #the following reshape should preserve the order of the indices further to the right
        #so the data extracted could be divided up into separate cells again.
        #hopefully using the same np.contatenate on all of these will do the same thing
        #fp = fp.reshape(num_gridpoints,fp.shape[-1])
        #xyz = xyz.reshape(num_gridpoints,xyz.shape[-1])

        #edensity = edensity.reshape(num_gridpoints,1)
        

        data['xyz'] = xyz
        data['fp'] = fp
        data['edensity'] = edensity
        if (self.include_init):
            init_edensity = init_edensity.reshape(num_gridpoints,1)
            data['init_edensity'] = init_edensity

        if (self.nspins>0):
            sdensity = sdensity.reshape(num_gridpoints,1)    
            data['sdensity'] = sdensity
            
            if (self.include_init):
                init_sdensity = init_sdensity.reshape(num_gridpoints,1)
                data['init_sdensity'] = init_sdensity
        
        good_data = True

        return data, good_data
        

        
        

            




