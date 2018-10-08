"""
Interface to regression and prediction
"""
from features.util import format_data,toy_argparse
import numpy as np
import tensorflow as tf
import pickle
import os
from features.heuristic_model import MLPGaussianRegressor,MLPDropoutGaussianRegressor
from features.bayes import vi_bayes
from edward import KLqp
from sklearn.metrics import mean_squared_error as mse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class regressor():
    def __init__(self,method="nonbayes",layers=[10,10],Nensemble=5,maxiter=5e3,activation="logistic",\
    batch_size=1.0,dtype=tf.float64,method_args={}):
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
        self.supp_methods = ["nonbayes","nonbayes_dropout","vi_bayes"]
        
        self.set_method(method)
        self.set_layers(layers)
        self.set_Nensemble(Nensemble)
        self.set_maxiter(maxiter)
        self.set_method_args(method_args)
        self.set_activation(activation)
        self.set_batch_size(batch_size)
        self.set_dtype(dtype)

    def set_method(self,method):
        """
        Set type of method to use
        """
        if method.lower() not in self.supp_methods: raise GeneralError("method {} not in {}".format(method,self.supp_methods))
        self.method = method.lower()
    
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
        self.Nensemble = Nensemble

    def set_maxiter(self,maxiter):
        """
        Set max iterations in optimization routine
        """
        self.maxiter = np.int32(maxiter)

    def set_method_args(self,method_args):
        """
        set method specific arguments
        """
        default_values = {"nonbayes":{"learning_rate":5e-3,"decay_rate":0.99,"alpha":0.5,"epsilon":1e-2,"grad_clip":100.0},\
                          "nonbayes_dropout":{"learning_rate":5e-3,"decay_rate":0.99,"alpha":0.5,"epsilon":1e-2,"grad_clip":100.0},\
                          "vi_bayes":{}}

        if any([_arg not in default_values[self.method].keys() for _arg in method_args.keys()]):
            raise GeneralError("unsupported key in {}".format(method_args.keys()))

        for _attr in default_values[self.method].keys():
            # use default is not given
            if _attr not in method_args.keys(): method_args.update({_attr:default_values[self.method][_attr]})
        
        self.method_args = method_args

    def set_activation(self,activation):
        """
        Set type of activation function to use for all nodes
        """
        if activation.lower() not in ["logistic","relu","tanh"]: raise GeneralError("activation type {} not supported".\
                format(activation))
        self.activation = activation.lower()                

    def set_batch_size(self,batch_size):
        """
        Set fraction of training observations to use (without replacement) in
        mini batch
        """
        if not isinstance(batch_size,(float,np.float,int,np.int)): raise GeneralError("invalid batch size {}".format(batch_size))
        self.batch_size = batch_size

    def set_dtype(self,dtype):
        """
        Set dtype for tf
        """
        self.dtype = dtype

    def fit(self,X,y):
        """
        Perform chosen regression on given data
        """
        
        # initial net weights assume 0 mean, 1 standard deviation
        self.train_data = format_data(X=X,y=y)
        
        # set mini batch size
        self.train_data.set_batch_size(self.batch_size)

        # feature dimensionality
        self.D = self.train_data.xs_standardized.shape[1]

        if self.method in  ["nonbayes","nonbayes_dropout"]:
            rmse = self._fit_nonbayes()
        elif self.method == "vi_bayes":
            rmse = self._fit_bayes()
        return rmse

    def predict(self,X,Nsample=None):
        """
        Make predictions. For VI Bayes methods, can improve predictive
        uncertainty by increasing number of samples drawn from posterior 
        predictive distribution - see method : set_Nensemble()

        Arguments
        ---------
        Nsample : int
            For bayes method. If given, the number of samples drawn from 
            approximate posterior distribution. Otherwise, self.Nensemble 
            samples are drawn
        """
        # use shift and scaling from training data
        xs_test = self.train_data.get_xs_standardized(X)

        if self.method in ["nonbayes","nonbayes_dropout"]:
            mean,var = self._predict_nonbayes(xs_test)
        elif self.method == "bayes":
            # numer of samples to draw from approx. posterior
            if Nsample is None: Nsample = self.Nensemble

            mean,var = self._predict_bayes(xs_test,Nsample)
        else: raise NotImplementedError

        return mean.flatten(),np.sqrt(var).flatten()

    def save(self,prefix="model"):
        """
        save everything necessary to make predictions later
        """
        if not os.path.isdir(prefix):
            os.mkdir('./{}'.format(prefix))

        if self.method in ["nonbayes","nonbayes_dropout"]:
            self._save_nonbayes(prefix)
        elif self.method == "bayes":
            self._save_bayes(prefix)

    def load(self,prefix="model"):
        """
        load prefiously saved model
        """
        if not os.path.isdir(prefix): raise GeneralError("Cannot find save directory {}".format(prefix))
        with open('{}/{}.pckl'.format(prefix,prefix),'rb') as f:
            attributes = pickle.load(f)
        f.close() 
        for _attr in attributes:
            # load all non tf attributes
            setattr(self,_attr,attributes[_attr])

        if self.method in ["nonbayes","nonbayes_dropout"]:
            self._load_nonbayes(prefix)
        elif self.method == "bayes":
            self._load_bayes(prefix)
    
    def _init_MLPGaussianRegressor(self):
        combine_args = self.method_args
        for _attr in ["activation","dtype"]:
            combine_args.update({_attr:getattr(self,_attr)})

        # a class with key,value as attribute name,value
        args = toy_argparse(combine_args)

        if self.method == "nonbayes":
            self.session["ensemble"] = [MLPGaussianRegressor(args, \
                    [self.D]+list(self.layers)+[2], 'model'+str(i)) for i in range(self.Nensemble)] 
        elif self.method == "nonbayes_dropout":
            self.session["ensemble"] = [MLPDropoutGaussianRegressor(args, \
                    [self.D]+list(self.layers)+[2], 'model'+str(i)) for i in range(self.Nensemble)] 


    def _fit_nonbayes(self):
        self.session = {"tf_session":None,"saver":None,"ensemble":None}
        
        # define tf variables
        self._init_MLPGaussianRegressor()

        self.session["tf_session"] = tf.Session()
        self.session["tf_session"].run(tf.global_variables_initializer())
        self.session["saver"] = tf.train.Saver(tf.global_variables())

        for model in self.session["ensemble"]:
            self.session["tf_session"].run(tf.assign(model.output_mean,self.train_data.target_mean))
            self.session["tf_session"].run(tf.assign(model.output_std,self.train_data.target_std))

        for itr in range(self.maxiter):
            for model in self.session["ensemble"]:
                # can train on distinct mini batches for each ensemble
                x,y = self.train_data.next_batch()

                feed = {model.input_data: x, model.target_data: y}
                _, nll, m, v = self.session["tf_session"].run([model.train_op, model.nll, model.mean, model.var], feed)

                if np.mod(itr,100)==0:
                    self.session["tf_session"].run(tf.assign(model.lr,\
                            self.method_args["learning_rate"]*(self.method_args["decay_rate"]**(itr/100))))
        
        # pass in standardized data
        pred_mean,pred_std = self._predict_nonbayes(self.train_data.xs_standardized)
        
        rmse = np.sqrt(mse(self.train_data.ys,pred_mean))
        return rmse

    def _predict_nonbayes(self,xs):
        en_mean = 0.0
        en_var = 0.0

        for model in self.session["ensemble"]:
            feed = {model.input_data: xs}
            mean,var = self.session["tf_session"].run([model.mean,model.var],feed)
            en_mean += mean
            en_var += var + mean**2
        
        en_mean /= len(self.session["ensemble"])
        en_var = en_var/len(self.session["ensemble"]) - en_mean**2
        return en_mean,en_var

    def _init_bayes(self):
        combine_args = self.method_args
        for _attr in ["activation","dtype"]:
            combine_args.update({_attr:getattr(self,_attr)})
        args = toy_argparse(combine_args)
        
        self.session["bayes_net"] = vi_bayes(args=args,layers=[self.D]+list(self.layers)) 

    def _fit_bayes(self):
        self.session = {"tf_session":None,"saver":None,"bayes_net":None}

        # initialise tf variables
        self._init_bayes() 

        rmse = self.session["bayes_net"].fit(X=self.train_data.xs_standardized,y=self.train_data.ys)
        return rmse


    def _predict_bayes(self,xs,Nsample):
        y_pred,y_std = self.session["bayes_net"].predict(xs,Nsample)

        return y_pred,y_std

    def _save_nonbayes(self,prefix):
        attributes = {}
        for _attr in [_a for _a in self.__dict__ if _a not in ["session"]]:
            attributes.update({_attr:getattr(self,_attr)})
        with open("./{}/{}.pckl".format(prefix,prefix),"wb") as f:
            pickle.dump(attributes,f)
        f.close()            
        self.session["saver"].save(self.session["tf_session"],"./{}/{}".format(prefix,prefix))

    def _save_bayes(self,prefix):
        raise NotImplementedError

    def _load_nonbayes(self,prefix):
        self.session = {"tf_session":None,"ensemble":None,"saver":None}
        self._init_MLPGaussianRegressor()
        self.session["tf_session"] = tf.Session()
        self.session["tf_session"].run(tf.global_variables_initializer())
        self.session["saver"] = tf.train.Saver(tf.global_variables())
        self.session["saver"].restore(self.session["tf_session"],"{}/{}".format(prefix,prefix))

    def _load_bayes(self,prefix):
        raise NotImplementedError

class GeneralError(Exception):
    pass
