import numpy as np
from features.powerspectrum_f90api import f90wrap_write_density_to_disk as f90_write_den
import scipy.linalg as linalg

class test_class():
    def __init__(self):
        pass

class format_data():
    def __init__(self,X=None,y=None,method='',val_fraction=0.05):
        """
        Pre processing involved for input to neural net
        """
        self.xs = None
        self.ys = None
        self.input_mean = None
        self.input_std = None
        self.xs_standardized = None
        self.target_mean = None
        self.target_std = None
        self.batch_size = None
        self.val_fraction = val_fraction
        self.val_idx=None
        self.batch = None

        single_target_methods = ["single_target_gaussian"]
        self.single_target = True if (method in single_target_methods) else False

        if X is not None and y is not None:
            self.set_data(X=X,y=y,val_frac=self.val_fraction)
        elif X is not None:
            self.set_data(X=X)
        elif y is not None:
            self.set_data(y=y)

    def set_data(self,X=None,y=None,val_frac=None):
        """
        transform X to have 0 mean, standard deviation of 1 in each dimension

        Attributes
        ----------
        X : shape=(N,D)
        y : shape=(N,D)
        """
        if X is not None:
            self.xs = X
            if (val_frac is not None):
                N_data = self.xs.shape[0]
                N_val = int(N_data*val_frac)
                if (self.val_idx is None):
                    self.val_idx = np.random.choice(np.arange(N_data),size=N_data,replace=False)
                self.xs_val = self.xs[self.val_idx[:N_val],:]
                self.xs = self.xs[self.val_idx[N_val:],:]

            self.input_mean = np.mean(self.xs,axis=0)
            self.input_std = np.std(self.xs,axis=0)

            if not isinstance(self.input_std,(list,np.ndarray)):
                # X is a float
                if self.input_std<1e-15:
                    self.input_std = 1.0
            else:
                # check for nonactive bases and avoid divide by zero
                idx = np.where(self.input_std<1e-15)[0]
                self.input_std[idx] = 1.0

            # 0 mean, 1 standard deviation
            self.xs_standardized = self.get_xs_standardized(self.xs)
        if y is not None:
            self.ys = y

            #self.target_std = np.std(self.ys,0)


            if (len(self.ys.shape)==1):
                self.y_dim = 1
                # need (N,1) rather than (N,) for all but the single target method
                if (not self.single_target):
                    self.ys = np.reshape(self.ys,(-1,1))
                    self.target_mean = np.mean(self.ys,0)
                    self.target_std = np.std(self.ys,0).reshape(-1,1)
                else:
                    self.target_mean = np.mean(self.ys)
                    self.target_std = np.std(self.ys)

            elif (self.ys.shape[1] == 1):
                self.y_dim = 1

                #this is the right shape for all but the single target method
                if (not self.single_target):
                    self.target_mean = np.mean(self.ys,0)
                    self.target_std = np.std(self.ys,0).reshape(-1,1)
                else:
                    self.ys = self.ys[:,0]
                    self.target_mean = np.mean(self.ys)
                    self.target_std = np.std(self.ys)



            else:
                assert (not self.single_target), "Method single_target_gaussian unable to train with multiple targets"
                self.y_dim = self.ys.shape[1]
                self.target_mean = np.mean(self.ys,0)
                self.target_std = linalg.sqrtm(np.cov(self.ys.T))

            if (val_frac is not None):
                N_data = self.ys.shape[0]
                N_val = int(N_data*val_frac)
                if (self.val_idx is None):
                    self.val_idx = np.random.choice(np.arange(N_data),size=N_data,replace=False)
                self.ys_val = self.ys[self.val_idx[:N_val],:]
                self.ys = self.ys[self.val_idx[N_val:],:]


    def set_batch_size(self,batch_size):
        """
        set size of mini batches as a fraction of total training observations
        """
        if batch_size<0.0 or batch_size>1.0: raise GeneralError("invalid batch size {} ".format(batch_size))
        self.batch_size = batch_size
        Ntrain = self.ys.shape[0]
        self.batch = int(Ntrain*self.batch_size)


    def get_xs_standardized(self,xs):
        """
        return standardized input
        """
        if self.input_mean is None or self.input_std is None: raise GeneralError("Must set x data before making predictions")
        res = (xs - self.input_mean)/self.input_std
        if len(res.shape)==1:
            # need (N,1) rather than (N,)
            res = np.reshape(res,(-1,1)).T
        return res

    def next_batch(self):
        """
        generate mini batch using self.batch_size
        """
        if self.batch_size is None: raise GeneralError("Batch size needs setting before mini batch generation")
        if self.xs_standardized is None or self.ys is None: raise GeneralError("Data must be set before generating batches")

        Ntrain = self.xs_standardized.shape[0]

        if np.isclose(self.batch_size,1):
            idx = np.arange(Ntrain)
        else:
            # choose without replacement, no repeats
            idx = np.random.choice(np.arange(Ntrain),size=int(self.batch_size*Ntrain),replace=False)

        return self.xs_standardized[idx],self.ys[idx]

    #ben's change
    def next_batch_set(self):
        """
        generate a set of mini batches using self.batch_size
        """
        if self.batch_size is None: raise GeneralError("Batch size needs setting before mini batch generation")
        if self.xs_standardized is None or self.ys is None: raise GeneralError("Data must be set before generating batches")

        Ntrain = self.xs_standardized.shape[0]

        idx = list(np.random.choice(np.arange(Ntrain),size=Ntrain,replace=False).astype(int))

        x_batch = []
        y_batch = []

        n_batches = int(1.0/self.batch_size)
        if (self.batch is None):
            self.batch = int(Ntrain*self.batch_size)

        for i in range(n_batches):
            x_batch.append(self.xs_standardized[idx[i*self.batch:(i+1)*self.batch],:])
            y_batch.append(self.ys[idx[i*self.batch:(i+1)*self.batch],:])


        return x_batch, y_batch

def tapering(x,xcut,scale):
    xprime = (x-xcut)/scale

    if not isinstance(x,(list,np.ndarray)):
        xprime = np.asarray([xprime])
    # places need to zero data
    idx = np.where(xprime>0.0)[0]

    xprime4 = xprime**4

    res = xprime4 / (1.0 + xprime4)
    res[idx] = 0.0

    if not isinstance(x,(list,np.ndarray)):
        res = res[0]
    return res

def write_density_to_disk(density,fft_grid,fname):
    f90_write_den(density,fft_grid[0],fft_grid[1],fft_grid[2],fname)

class toy_argparse():
    def __init__(self,args):
        """
        mimick behaviour of argparse.parse_args() instance
        """
        if not isinstance(args,dict): raise GeneralError("arg to toy_argparse() must be a dictionary")

        for _key in args.keys():
            # assign key,value as attribute name,values
            setattr(self,_key,args[_key])

class GeneralError(Exception):
    pass
