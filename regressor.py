"""
Interface to regression and prediction
"""
from features.util import format_data
import numpy as np
import tensorflow as tf

class regressor():
    def __init__(self,method="heuristic",layers=[10,10],Nensemble=5,maxiter=5e3,method_args={}):
        """
        Interface to regression using heuristic ensembles or VI Bayes. In both
        cases, uncertainties are made by an ensemble of nets or repeated 
        sampling of posterior predictive distribution in the case of Bayes.

        Detailed outline of methods can be found
        
        (heuristic) : arXiv:1612.01474, Simple and Scalable Predictive 
        Uncertainty Estimation using Deep Ensembles, (2017)

        (VI bayes) : arXiv:1610.09787, Edward: A library for probabilistic 
        modeling, inference, and criticism
        """
        self.supp_methods = ["heuristic","vi_bayes"]
        
        self.set_method(method)
        self.set_layers(layers)
        self.set_Nensemble(Nensemble)
        self.set_maxiter(maxiter)
        self.set_method_args(method_args)

    def set_method(self,method):
        """
        Set type of method to use
        """
        if method.lower() not in self.supp_methods: raise GeneralError("method {} not in {}".format(method,self.supp_methods))
        self.method = self.supp_methods
    
    def set_layers(self,layers):
        """
        Set number of nodes in each layer
        """
        if not isinstance(layers,(np.ndarray,list)): raise GeneralError("layers type {} not supported".format(type(layers)))
        self.layers = np.asarray(layers,dtype=int)
        
    def set_Nensemble(self,Nensemble):
        """
        Set either size of ensemble in heuristic approach, or number of samples
        drawn from posterio predictive distribution in Bayes approach
        """
        if not isinstance(Nensemble,int) or Nensemble<1: raise GeneralError("invalid ensemble size {}".format(Nensemble))

    def set_maxiter(self,maxiter):
        """
        Set max iterations in optimization routine
        """
        self.maxiter = np.int32(maxiter)

    def set_method_args(self,method_args):
        """
        set method specific arguments
        """
        default_values = {"heuristic":{"learning_rate":5e-3,"decay_rate":0.99,"alpha":0.5,"epsilon":1e-2},\
                          "vi_bayes":{}}

        if any([_arg not in default_values[self.method].keys() for _arg in method_args.keys()]):
            raise GeneralError("unsupported key in {}".format(method_args.keys()))

        for _attr in default_values:
            # use default is not given
            if _attr not in method_args.keys(): method_args.update({_attr:default_values[self.method][_attr]})
        
        self.method_args = method_args

    def fit(self,X,y):
        """
        Perform chosen regression on given data
        """
        
        # initial net weights assume 0 mean, 1 standard deviation
        self.train_data = format_data(X=X,y=y)

    def predict(self,X):
        """
        Make predictions. For VI Bayes methods, can improve predictive
        uncertainty by increasing number of samples drawn from posterior 
        predictive distribution - see method : set_Nensemble()
        """
        # use shift and scaling from training data
        xs_test = self.train_data.get_xs_standardized(X)

        if self.method == "heuristic":
            mean,var = self._predict_heuristic(xs_test)
        else:
            mean,var = self._predict_bayes(xs_test)

    def _fit_heuristic(self):
        ensemble = [MLPGaussianRegressor(args, self.layers, 'model'+str(i)) for i in range(self.Nensemble)] 
    
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        for model in ensemble:
            self.session.run(tf.assign(model.output_mean,self.train_data.target_mean))
            self.session.run(tf.assign(model.output_std,self.train_data.target_std))

        for itr in range(self.max_iter):
            for model in ensemble:
                # can train on distinct mini batches for each ensemble
                x,y = self.train_data.next_batch()

                feed = {model.input_data: x, model.target_data: y}
                _, nll, m, v = self.session.run([model.train_op, model.nll, model.mean, model.var], feed)

                if np.mod(itr,100)==0:
                    self.session.run(tf.assign(model.lr,\
                            self.method_args["learning_rate"]*(self.method_args["decay_rate"]**(itr/100))))
    
        self.session = {"tf_session":self.session,"ensemble":ensemble}

    def _predict_heuristic(self,xs):
        en_mean = 0.0
        en_var = 0.0

        for model in self.session["ensemble"]:
            feed = {model.input_data: xs}
            mean,var = sess.run([model.mean,model.var],feed)
            en_mean += mean
            en_var += var + mean**2
        
        en_mean /= len(ensemble)
        en_var = en_var/len(ensemble) - en_mean**2
        return en_mean,en_var

    def _fit_bayes(self):
        raise NotImplementedError
    def _predict_bayes(self,xs):
        raise NotImplementedError
        return None,None

class GeneralError(Exception):
    pass
