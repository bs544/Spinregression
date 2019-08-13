import numpy as np
from parsers.dft.parser_castep import parse
#parser code courtesy of Andrew Fowler
from parsers.structure_class import supercell
from energyregression.rad_fp import calculate
from energyregression.EAMcalc import get_EAM_energies
from features.regressor import regressor
from features.bispectrum import calculate as spin_calculate
import pickle
import os
import copy
import time

class Cell_data():
    def __init__(self,datadir,savedir,verbose=False,savename='cell_data',spin=True):
        self.datadir = datadir
        self.savedir = savedir
        self.verbose = verbose
        self.savename = savename
        self.data = []
        self.spin = spin
        self.keys = ['cell','positions','forces','energy']
        if (self.spin):
            self.keys.append('spin')

    def get_spins_from_castep(self,filename,natoms):
        """
        Parameters:
            filename: (str) name of the .castep file to examine
            natoms: (int) number of atoms, this has to be the same as the length of the array
        Returns:
            data: (array) shape (N,) Mulliken Analysis of spin contribution from each atom in units of hbar/2
        """
        assert (filename[-7:]==".castep"), "Incorrect file type submitted to get_spins_from_castep, needs to be .castep, not {}".format(filename[-7:])

        with open(filename,'r') as f:
            lines = f.readlines()
        
        spin_calc = True

        for line in lines:
            if ('treating system as non-spin-polarized' in line):
                spin_calc = False
                break
        if (not spin_calc):
            if (self.verbose):
                print('Spin independent calculations, returning None as spin')
            return None

        data = []
        for line in lines:
            if ('  Fe              ' in line):
                data.append(float(line.split()[-1]))
        data = np.asarray(data)
        assert (len(data)==natoms), "data isn't the right size"
        return data

    def get_init_spins_from_castep(self,filename,natoms):
        """
        Parameters:
            filename: (str) name of the .castep file to examine
            natoms: (int) number of atoms, this has to be the same as the length of the array
        Returns:
            data: (array) shape (N,) Initial spin contribution from each atom in units of hbar/2
        """
        assert (filename[-7:]==".castep"), "Incorrect file type submitted to get_spins_from_castep, needs to be .castep, not {}".format(filename[-7:])

        with open(filename,'r') as f:
            lines = f.readlines()
        
        spin_calc = True

        for line in lines:
            if ('treating system as non-spin-polarized' in line):
                spin_calc = False
                break
        if (not spin_calc):
            if (self.verbose):
                print('Spin independent calculations, returning None as spin')
            return None

        data = []
        line_num = 0
        for i,line in enumerate(lines):
            if ('Initial magnetic' in line):
                line_num = i
        try:
            assert (line_num != 0), "Can't find initial magentic moment"
        except AssertionError as error:
            if (self.verbose):
                print(error)
            return None
        lines = lines[line_num+3:line_num+3+natoms]
        for line in lines:
            split = line.split()
            data.append(int(float(split[-2])))
        data = np.asarray(data)
        return data

    def load_castep(self,filename):
        """
        Parameters:
            file: (str) name of the .castep file to load
        Returns:
            data: (dict) dictionary containing the following keys:
                    'energy': energy of the cell in eV
                    'cell': (3,3) array of cell vectors in Angstroms
                    'forces': (N,3) array of forces in eV/Angstrom
                    'positions': (N,3) array of positions in Angstroms
                    'spin': (N,) array of Mulliken spins (hbar/2)
            status: (bool) if it returns false, then the load has failed
        """

        if ('/' not in filename):
            filename = '{}{}'.format(self.datadir,filename)
            if (self.verbose):
                print('No directory specified, assuming the desired file is in {}'.format(self.datadir))

        
        ftype = 'castep'
        l = len(ftype)

        try:
            assert ftype in filename[-l:], 'need to pass a .castep file to the get_positions() function of the parser'
        except AssertionError as error:
            if (self.verbose):
                print(error)
            return None, False
        data = {}

        cas_parser = parse(filename,ftype)
        cas_parser.run()
        cas_cell = cas_parser.get_supercells()[0]

        posns = cas_cell.get_positions()
        species = cas_cell.get_species()
        cell = cas_cell.get_cell()
        forces = cas_cell.get_forces()
        energy = cas_cell.get_energy()

        #positions are fractional coordinates when they come out
        #multiply by cell values to get coordinates in Angstroms
        for i in range(posns.shape[0]):
            posns[i,:] = np.dot(cell,posns[i,:])

        data['cell'] = cell
        data['elements'] = species
        data['positions'] = posns
        data['forces'] = forces
        data['energy'] = energy

        if (self.spin):

            spins = self.get_spins_from_castep(filename,posns.shape[0])
            init_spins = self.get_init_spins_from_castep(filename,posns.shape[0])
            data['spin'] = spins
            data['init_spins'] = init_spins
 
        return data, True
    
    def load_castep_dir(self,dir_=None,number=None):
        """
        Parameters
            dir: (str) if this isn't none, then this is used as the directory from which .castep files are parsed
            number: (int) number of files to load, 
                if this is None or the number of files in the directory is less than this number, the all the files are considered
        Returns:
            datalist: (list) each element is a dictionary with the following keys: 'energy','cell','forces','spin','positions'
            status: (bool) whether any useable data was loaded
        """

        if (dir_ is not None):
            datadir = dir_
        else:
            datadir = self.datadir

        assert os.path.isdir(datadir), "Specified directory: {} is not a directory".format(datadir)
        
        files = sorted(os.listdir(datadir))

        if (number is not None):
            num_files = number
        else:
            num_files = len(files)
        
        datalist = []
        cntr = 0


        for file_ in files:
            filename = '{}{}'.format(datadir,file_)
            data, goodload = self.load_castep(filename)
            if (goodload):
                cntr += 1
                datalist.append(data)
                if (cntr == num_files):
                    if (self.verbose):
                        print('{} files loaded'.format(cntr))
                    return datalist, True
        
        if (self.verbose):
            print('{} files loaded out of a possible {}'.format(cntr,len(files)))
        
        if (len(datalist)>0):
            status = True
        else:
            status = False

        return datalist, status

    def load_pickle_dir(self,pickle_dir=None):
        """
        Parameters:
            pickle_dir: (str) directory of pickle files, if None then use self.savedir
        Returns:
            datalist: (list) each element is a dictionary with the following keys: 'energy','cell','forces','spin','positions'
                    These are loaded from the .pckl files, which is quicker than getting them from the .castep files
            status: (bool) if the load goes well
        """
        
        if (pickle_dir is None):
            pickle_dir = self.savedir
        
        try:
        
            assert os.path.isdir(pickle_dir), "{} is not a directory".format(pickle_dir)
        except AssertionError as error:
            if (self.verbose):
                print(error)
            return None, False
        
        files = sorted(os.listdir(pickle_dir))

        datalist = []

        for file_ in files:
            if (".pckl" not in file_):
                continue
            filename = '{}{}'.format(pickle_dir,file_)

            with open(filename,'rb') as f:
                saved_dict = pickle.load(f)

                try:
                    # and all(key in self.keys for key in saved_dict.keys()),
                    assert all(key in saved_dict.keys() for key in self.keys), "Saved dictionary has an incompatible set of keys\n saved keys: {} \n expected keys: {}".format(saved_dict.keys(),self.keys)
                except AssertionError as error:
                    if (self.verbose):
                        print(error)
                        print('Skipping file {}'.format(file_))
                    continue
            
            datalist.append(saved_dict)
        
        if (len(datalist)> 0):
            if (self.verbose):
                print('{} files loaded out of a possible {} in the directory'.format(len(datalist),len(files)))
            return datalist, True
        else:
            if (self.verbose):
                print('No files loaded')
            return datalist, False
    
    def save_data(self,datalist,savedir=None,savename=None):
        """
        Parameters:
            datalist: (list) elements are dictionaries to be saved
            savedir: (str) directory to which the .pckl files are saved, if None then self.savedir is used
            savename: (str) base name given to the .pckl files, if None then self.savename is used
        Actions:
            saves the dictionaries in the list as individual pickle files
        Returns:
            status: (bool) whether the save was good
        """
        
        if (savedir is None):
            savedir = self.savedir
        
        if (savename is None):
            savename = self.savename
        
        if (not os.path.isdir(savedir)):
            os.mkdir(savedir)
            if (self.verbose):
                print("{} doesn't exist, creating directory".format(savedir))
        
        for i, data in enumerate(datalist):
            filename = '{}{}_{}.pckl'.format(savedir,savename,i)

            if (self.verbose):
                print("Saving castep data into {}".format(filename))
            
            with open(filename,'wb') as f:
                pickle.dump(data,f)
        return True
    
    def load_cell_data(self,load=True,save=True,datadir=None,savedir=None,cell_list=None):
        """
        Parameters:
            load: (bool) attempt to load existing pickle data if True
            save: (bool) if no pickle data exists, save the parsed data in this form if True
            datadir: (str) The directory from which .castep files are parsed, if None then self.datadir is used
            savedir: (str) directory to which the .pckl files are saved, if None then self.savedir is used
            cell_list: (list) contains list of data indices to use
                      (int)  contains number of files to parse
                      If None, then parse all of the files
        Actions:
            loads all of the data specified
        """
        start = time.time()

        if (savedir is None):
            savedir = self.savedir
        if (datadir is None):
            datadir = self.datadir

        if (load):
            if (self.verbose):
                print('Attempting to load data')
            datalist,goodload = self.load_pickle_dir(pickle_dir=savedir)
            if (goodload):
                if (self.verbose):
                    print('Good Load')
                self.data = datalist
                return
            elif (self.verbose):
                print('Failed to load, will read data from {} instead'.format(datadir))
        
        datalist,goodload = self.load_castep_dir(dir_=datadir)

        if (goodload):
            if (self.verbose):
                print('Data parsed successfully')
            self.data = datalist
        elif (not goodload and self.verbose):
            print('Data not parsed successfully, Aborting')
            return
        
        if (save):
            well_saved = self.save_data(self.data,savedir=savedir)
            if (well_saved and self.verbose):
                print('Data saved successfully')
            elif (not well_saved and self.verbose):
                print('Data not saved successfully')
        
        if (self.verbose):
            print('Time taken: {0:.2f}s'.format(time.time()-start))
        return
            
    def write_potfit_configs(self,write_dir,filename='Potfit_configs.txt',cell_list=None,useforce=True,append=True):
        """
        Parameters:
            cell_list: (list) list of cell indices from self.data to use, if None then use all values
            write_dir: (str) directory in which to write the configurations file
            append: (bool) if True then will append the potfit if it already exists
        Actions:
            Assumes you have loaded the data already so self.data can be used
            writes configuration file that can be used by potfit
        overall look of each configuration:
        #N natoms useforce
        #X boxx.x boxx.y boxx.z
        #Y boxy.x boxy.y boxy.z
        #Z boxz.x boxz.y boxz.z
        #E coh_eng
        #F 
        0 x y z f_x f_y f_z
        0 x y z f_x f_y f_z
        0 x y z f_x f_y f_z
        0 x y z f_x f_y f_z

        these configurations are concatenated in the same config file
        coh_energy is the energy/#atoms - individual atom energy in this case
        """

        Fe_atom_energy = -853.8262 #eV
        

        filename = '{}{}'.format(write_dir,filename)
        if (append):
            f = open(filename,'a')
        else:
            f = open(filename,'w')

        force_num = 1 if useforce else 0

        if (cell_list is None):
            cell_list = [i for i in range(len(self.data))]
        elif (isinstance(cell_list,int)):
            cell_list = [i for i in range(cell_list)]

        for i in cell_list:
            data = self.data[i]
            positions = data['positions']
            energy = data['energy']
            forces = data['forces']
            cell_vect = data['cell']

            natoms = positions.shape[0]
            coh_energy = energy/natoms - Fe_atom_energy

            f.write('#N {} {}\n'.format(natoms,force_num)) #1 to indicate that force should be used
            f.write('#X\t{0:.5f}\t{1:.5f}\t{2:.5f}\n'.format(cell_vect[0,0],cell_vect[0,1],cell_vect[0,2]))
            f.write('#Y\t{0:.5f}\t{1:.5f}\t{2:.5f}\n'.format(cell_vect[1,0],cell_vect[1,1],cell_vect[1,2]))
            f.write('#Z\t{0:.5f}\t{1:.5f}\t{2:.5f}\n'.format(cell_vect[2,0],cell_vect[2,1],cell_vect[2,2]))
            f.write('#E {}\n'.format(coh_energy))
            f.write('#F\n')
            for i in range(natoms):
                f.write('0 {0:.9f}\t{1:.9f}\t{2:.9f}\t{3:.5f} \t{4:.5f} \t{5:.5f}\n'.format(positions[i,0],positions[i,1],positions[i,2],forces[i,0],forces[i,1],forces[i,2]))
        
        f.close()

    def get_weights_from_net(self,fp_dict,cell,positions,net,init_weights):
        """
        Parameters:
            fp_dict: (dict) contains the parameters for the network input: nmax,lmax,rcut,local_fptype,global_fptype,gnmax,glmax,grcut
            cell: (array) shape (3,3) cell vectors
            positions: (array) shape (N,3) fractional coordinates of the atoms
            name: (str) name of the network, needed to load it
            init_weights: (list) initial weights for the fingerprints
        Returns:
            weightings: (list) weightings of atoms based on predicted spins
        """
        
        fp = spin_calculate(cell,positions,positions,fp_dict['nmax'],fp_dict['lmax'],fp_dict['rcut'],local_form=fp_dict['local_form'],\
            global_form=fp_dict['global_form'],gnmax=fp_dict['gnmax'],glmax=fp_dict['glmax'],grcut=fp_dict['grcut'],weighting=init_weights)

        spin = net.predict(fp)
        weights = list(0.5*spin+4)# so it becomes number of spin up valence electrons
        return weights

    def get_data_array(self,nmax=4,weighting=None,rcut=6.0,parallel=True,netname=None,net_dict=None,cohesive=True,EAMfile=None):
        """
        Parameters:
            nmax: number of radial basis functions to use in radial descriptor
            weighting: (list) weightings on atoms, if None, then use the atomic spins
            rcut: (float) cutoff radius for radial descriptions
            parallel: (bool) whether to parallelise calculations
            net_name: (str) name of the network, if you don't want network predicted spins, set this to None
            net_dict: (str) dictionary of network parameters, again set to None if you don't want to use the network
            cohesive: (bool) whether to use the energy directly from castep, or the cohesive energy per atom
            EAMfile: (str) name of file containing EAM parameters. If this is not None, the eam predicted energy is subtracted from the cohesive energy
        Returns:
            decription_data: (array) shape (Ncells,NatomsperCell,1) gives the atomic spins in this form
            energy_data: (array) shape (Ncells,1) gives the cell energy in this form
        """
        natoms = []
        spins  = []
        energies = []
        Fe_energy = -853.8262#eV
        if (EAMfile is not None):
            cohesive = True
        for datadict in self.data:
            natoms.append(len(datadict['spin']))
            spins.append(datadict['spin'])
            energies.append(datadict['energy'])
        
        NatomsperCell = max(natoms)
        description_data = np.zeros((len(self.data),NatomsperCell,2+nmax))
        energy_data = np.zeros((len(self.data),1))

        if (netname is not None):
            net = regressor()
            net.load(netname)
        cell_list = []
        position_list = []
        for i in range(len(self.data)):
            description_data[i,:,0] = spins[i]
            description_data[i,:,1] = np.tile(sum(spins[i])/len(spins[i]),len(spins[i]))

            cell = self.data[i]['cell']
            positions = self.data[i]['positions']
            cell_list.append(cell)
            position_list.append(positions)
            positions = np.dot(positions,np.linalg.inv(cell))
            if (weighting is None):
                if (netname is not None and net_dict is not None):
                    init_weights = self.data[i]['init_spin']
                    weighting_ = self.get_weights_from_net(net_dict,cell,positions,net,init_weights)
                else:
                    weighting_ = list(0.5*self.data[i]['spin'] + 4)
            else:
                weighting_ = weighting
            description_data[i,:,2:] = calculate(cell,positions,nmax,rcut=rcut,parallel=parallel,weighting=weighting_)

            if (cohesive):
                energy_data[i,0] = energies[i]/positions.shape[0] - Fe_energy
            else:
                energy_data[i,0] = energies[i]
        if (EAMfile is not None):
            eam_energy = get_EAM_energies(cell_list,position_list,EAMfile)
            #only works because all of the cells have the same number of atoms
            eam_energy = np.array(eam_energy).reshape(-1,1)*0.25
            energy_data -= eam_energy
        return description_data, energy_data



# dir_ = '../../Castep_Data/Spin/MultiConfigFCC/'
# handler = Cell_data(dir_,'./Pickle_Data/',verbose=True)
# handler.load_cell_data(save=False,load=False)
# idx = np.random.choice(np.arange(len(handler.data)),750,replace=False)
# cell_list = [i for i in idx[:750]]
# handler.write_potfit_configs('./',filename='MixedFCCconfigs.txt',cell_list=cell_list)
