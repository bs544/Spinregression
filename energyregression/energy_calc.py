import numpy as np
from network import Cell_network
import os
import pickle

class energy_calculator():
    def __init__(self,net_arg_dict,regression_arg_dict,layers):
        self.net_arg_dict = net_arg_dict
        self.regression_arg_dict = regression_arg_dict
        self.layers = layers
        # self.setup_network()
        self.network = None
        self.use_regression_arg_dict(regression_arg_dict)
    
    def use_regression_arg_dict(self,regression_arg_dict):
        """
        Parameters:
            regression_arg_dict: (dict) arguments for regression
        Actions:
            sets arguments to be class properties
        """
        self.batch_frac = regression_arg_dict['batch_size']
        self.decay_step = regression_arg_dict['decay_step']
        self.decay_rate = regression_arg_dict['decay_rate']
        self.val_frac = regression_arg_dict['val_frac']
        self.check_interval = regression_arg_dict['check_interval']
        self.niter = regression_arg_dict['niter']
    
    def setup_network(self,layers=None):
        if (layers is None):
            layers = self.layers
        self.network = Cell_network(self.net_arg_dict,layers,'energyregressor')
    
    def prepare_train_data(self,x,y):
        """
        Parameters:
            x: (array) shape (Ncells,NatomsperCell,fplength) training input
            y: (array) shape (Ncells,1) training output
        Actions:
            sets up the input mean and standard deviation so data can be centered and standardised before being used
        Returns:
            train_x: (array) shape (Ncells,NatomsperCell,fplength) centered and standardised NN input
            train_y: (array) shape (Ncells,1) centered and standardised target data
        """

        #To Do: sort this out for a variable number of atoms in the cell
        
        flat_x = x.reshape(-1,x.shape[-1])
        self.input_mean = np.mean(flat_x,axis=0)
        self.input_std = np.std(flat_x,axis=0)

        self.output_mean = np.mean(y)
        self.output_std = np.std(y)

        self.Ntrain = x.shape[0]

        idx = list(np.random.choice(np.arange(self.Ntrain),size=self.Ntrain,replace=False).astype(int))

        val_idx = idx[:int(self.val_frac*self.Ntrain)]
        train_idx = idx[int(self.val_frac*self.Ntrain):]

        train_y = (y-self.output_mean)/self.output_std
        train_x = (x-self.input_mean)/self.input_std

        self.val_x = train_x[val_idx,:,:]
        self.val_y = train_y[val_idx,:]
        train_x = train_x[train_idx,:,:]
        train_y = train_y[train_idx,:]

        self.Ntrain = train_x.shape[0]
        self.batch_size = int(self.Ntrain*self.batch_frac)

        return train_x,train_y
    
    def get_batches(self,x,y):
        """
        Parameters:
            x: (array) shape (Ncells,NatomsperCell,fplen) NN input
            y: (array) shape (Ncells,1) NN target data
        Returns:
            x_batches: (list) set of floor(self.Ntrain/self.batch_size) batches as elements
            y_batches: (list) same set, but for the y data
        """

        self.n_batches = int(y.shape[0]/self.batch_size)
        x_batch = []
        y_batch = []

        idx = list(np.random.choice(np.arange(self.Ntrain),size=self.Ntrain,replace=False).astype(int))

        for i in range(self.n_batches):            
            x_batch.append(x[idx[i*self.batch_size:(i+1)*self.batch_size],:,:])
            y_batch.append(y[idx[i*self.batch_size:(i+1)*self.batch_size],:])
        
        return x_batch,y_batch
    
    def fit(self,x,y):
        """
        Parameters:
            x: (array) shape (Ncells,NatomsperCell,fplength) training input
            y: (array) shape (Ncells,1) training output
        Actions:
            parameterises a neural network on energy data for cells
        """

        self.net_layers = [x.shape[-1]]+ self.layers + [y.shape[-1]]

        self.setup_network(layers=self.net_layers)

        train_x, train_y = self.prepare_train_data(x,y)

        self.n_batches = int(train_y.shape[0]/self.batch_size)
        
        self.loss = []
        self.val_loss = []

        nepochs = int(self.niter/self.n_batches)

        cntr = 0
        for i in range(nepochs):
            x_batch, y_batch = self.get_batches(train_x,train_y)

            for j in range(self.n_batches):
                loss_ = self.network.train_step(x_batch[j],y_batch[j])

                if (cntr%self.check_interval == 0):
                    self.loss.append(loss_)
                    val_pred = self.network.predict(self.val_x)
                    self.val_loss.append(np.sqrt(np.mean(np.square(val_pred-self.val_y))))
                cntr += 1

                if (cntr%self.decay_step == 0):
                    lr = self.net_arg_dict['lr']
                    lr *= (self.decay_rate)**(cntr/self.decay_step)
                    self.network.update_lr(lr)
        
        self.loss = np.asarray(self.loss)
        self.val_loss = np.asarray(self.val_loss)
                
        return

    def predict_energies(self,x):
        """
        Parameters:
            x: (array) shape (Ncells,NatomsperCell,fplength)
        Returns:
            Energies: (array) shape (Ncells,1)
        """

        input_x = (x-self.input_mean)/self.input_std
        net_out = self.network.predict(input_x)
        Energies = (net_out*self.output_std)+self.output_mean
        return Energies
    
    def save(self,name='model'):
        """
        Parameters:
            name: (str) name to give to the saved network
        Actions:
            saves the network in a directory called model so that it can be loaded and used again later
        """
        if (not os.path.isdir(name)):
            os.mkdir('./{}'.format(name))
        self.network.save_network(name)
        attributes = {}
        for _attr in [_a for _a in self.__dict__ if _a not in ['network']]:
            attributes.update({_attr:getattr(self,_attr)})
        with open("./{}/calc_{}.pckl".format(name,name),"wb") as f:
            pickle.dump(attributes,f)
        
    
    def load(self,name='model'):
        """
        Parameters:
            name: (str) name of the saved network
        Actions:
            loads the saved network and the dictionary of class parameters for the energy calculator and the network class
        """
        assert (os.path.isdir(name)), "Cannot find save directory {}".format(name)
        with open('{}/calc_{}.pckl'.format(name,name),'rb') as f:
            attributes = pickle.load(f)
        for _attr in attributes:
            # load all non tf attributes
            setattr(self,_attr,attributes[_attr])
        if (self.network is None):
            self.setup_network(layers = self.net_layers)
        self.network.load_network(name)
        
        





