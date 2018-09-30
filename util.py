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

    def set_data(X=None,y=None):
        """
        transform X to have 0 mean, standard deviation of 1 in each dimension
        
        Attributes
        ----------
        X : shape=(N,D)
        y : shape=(D,)
        """
        if X is not None:
            self.xs = X
            self.input_mean = np.mean(self.xs,axis=0)
            self.input_std = np.std(self.xs,axis=0)
            
            # 0 mean, 1 standard deviation
            self.xs_standardized = self.get_xs_standardized(self.xs)
        if y is not None:
            self.ys = y
            self.target_mean = np.mean(self.ys,0)[0]
            self.target_std = np.std(self.ys,0)[0]

    def get_xs_standardized(self,xs):
        """
        return standardized input
        """
        if self.input_mean is None or self.input_std is None: raise GeneralError("Must set x data before making predictions")
        return (xs - self.input_mean)/self.input_std



class GeneralError(Exception):
    pass
