import numpy as np

class format_data():
    def __init__(self,X=None,y=None):
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

        if X is not None:
            self.set_data(X=X)
        if y is not None:
            self.set_data(y=y)

    def set_data(self,X=None,y=None):
        """
        transform X to have 0 mean, standard deviation of 1 in each dimension
        
        Attributes
        ----------
        X : shape=(N,D)
        y : shape=(N,D)
        """
        if X is not None:
            self.xs = X
            self.input_mean = np.mean(self.xs,axis=0)
            self.input_std = np.std(self.xs,axis=0)
            
            # 0 mean, 1 standard deviation
            self.xs_standardized = self.get_xs_standardized(self.xs)

            if len(self.xs_standardized.shape)==1:
                # need (N,1) rather than (N,)
                self.xs_standardized = np.reshape(self.xs_standardized,(-1,1))
        if y is not None:
            self.ys = y
            self.target_mean = np.mean(self.ys,0)
            self.target_std = np.std(self.ys,0)
            
            if len(self.ys.shape)==1:
                self.ys = np.reshape(self.ys,(-1,1))

    def set_batch_size(self,batch_size):
        """
        set size of mini batches as a fraction of total training observations
        """
        if batch_size<0.0 or batch_size>1.0: raise GeneralError("invalid batch size {} ".format(batch_size))
        self.batch_size = batch_size
        

    def get_xs_standardized(self,xs):
        """
        return standardized input
        """
        if self.input_mean is None or self.input_std is None: raise GeneralError("Must set x data before making predictions")
        return (xs - self.input_mean)/self.input_std


    def next_batch(self):
        """
        generate mini batch using self.batch_size
        """
        if self.batch_size is None: raise GeneralError("Batch size needs setting before mini batch generation")
        if self.xs_standardized is None or self.ys is None: raise GeneralError("Data must be set before generating batches")

        Ntrain = self.xs_standardized.shape[0]

        # choose without replacement, no repeats
        idx = np.random.choice(np.arange(Ntrain),size=int(self.batch_size*Ntrain),replace=False)

        return self.xs_standardized[idx],self.ys[idx]

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
