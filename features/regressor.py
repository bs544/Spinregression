"""
Interface to regression and prediction
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_AFFINITY'] = 'noverbose'


from features.util import format_data,toy_argparse
import pickle

from features.heuristic_model import MLPGaussianRegressor,NoLearnedCovariance,SingleTargetGaussianRegressor
from sklearn.metrics import mean_squared_error as mse
import copy
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import numpy as np
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)



class regressor():
    def __init__(self,method="nonbayes",layers=[10,10],Nensemble=5,maxiter=5e3,activation="logistic",\
    batch_size=1.0,dtype=tf.float64,method_args={},load=None):
        """
        Interface to regression using heuristic ensembles. Uncertainties are made by an ensemble of nets

        Detailed outline of the method can be found

        (heuristic) : arXiv:1612.01474, Simple and Scalable Predictive
        Uncertainty Estimation using Deep Ensembles, (2017)
        """
        self.supp_methods = ["nonbayes","standard_ensemble","single_target_gaussian"]

        if load is None:
            self.set_method(method)
            self.set_layers(layers)
            self.set_Nensemble(Nensemble)
            self.set_maxiter(maxiter)
            self.set_method_args(method_args)
            self.set_activation(activation)
            self.set_batch_size(batch_size)
            self.set_dtype(dtype)
        else:
            self.load(load)

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
        Set size of ensemble in heuristic approach
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
        gradient clipping caps the value of the gradient used by the optimiser to avoid gradient explosion
        """
        default_values = {"nonbayes":{"learning_rate":5e-3,"decay_rate":0.99,"grad_clip":100.0,"learn_inverse":True},\
                          "standard_ensemble":{"learning_rate":5e-3,"grad_clip":100.0,"decay_rate":0.99},\
                          "single_target_gaussian":{"learning_rate":5e-3,"grad_clip":100.0,"decay_rate":0.99}}

        for _method in ["nonbayes","standard_ensemble","single_target_gaussian"]:
            default_values[_method].update({"opt_method":"rmsprop"})

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
        # feed in method as well so that training data need only by scalars for single target data
        self.train_data = format_data(X=X,y=y,method=self.method)
        self.y_dim = self.train_data.y_dim

        # set mini batch size
        self.train_data.set_batch_size(self.batch_size)

        # feature dimensionality
        self.D = self.train_data.xs_standardized.shape[1]

        rmse, rmse_val = self._fit_nonbayes()
        
        return rmse, rmse_val

    def predict(self,X):
        """
        Make predictions.

        Arguments
        ---------
        X: input data that has yet to be centred and standardised
        
        Returns
        ---------
        mean of precictions
        predicted standard deviation
        """
        # use shift and scaling from training data
        xs_test = self.train_data.get_xs_standardized(X)

        if self.method in ["nonbayes","standard_ensemble","single_target_gaussian"]:
            mean,var = self._predict_nonbayes(xs_test)
        else: raise NotImplementedError

        return mean,np.sqrt(var)
    
    def predict_individual_values(self,X,model_idx=None):
        """
        Make predictions for individual networks in the ensemble, only returns means for RMSE calculations
        Only implemented for the standard-ensemble and nonbayes ensembles

        model_idx specifies which network in the ensemble to use, if it's None, then get the data for all of the values
        """

        #X has shape [num_values,fingerprint_length]

        xs = self.train_data.get_xs_standardized(X)

        num_values = xs.shape[0]

        means = np.zeros((self.Nensemble,num_values,self.train_data.y_dim))

        if (model_idx is None):
            model_list = [i for i in range(len(self.session['ensemble']))]
        else:
            model_list = [model_idx]

        for ii,model in enumerate([self.session['ensemble'][idx] for idx in model_list]):
            feed = {model.input_data: xs}

            if self.method == 'nonbayes':
                mean = self.session["tf_session"].run([model.mean],feed)
                means[ii,:,:] = mean


            elif self.method == "standard_ensemble":
                mean = self.session["tf_session"].run([model.mean],feed)
                means[ii,:,:] = mean[0]
            
            elif self.method == "single_target_gaussian":
                mean = self.session["tf_session"].run(model.mean,feed)
                means[ii,:,:] = mean
        
        return means

    def save(self,prefix="model"):
        """
        save everything necessary to make predictions later
        """
        if not os.path.isdir(prefix):
            os.mkdir('./{}'.format(prefix))

        self._save_nonbayes(prefix)

    def load(self,prefix="model"):
        """
        load previously saved model
        """
        if not os.path.isdir(prefix): raise GeneralError("Cannot find save directory {}".format(prefix))
        with open('{}/{}.pckl'.format(prefix,prefix),'rb') as f:
            attributes = pickle.load(f)
        f.close()
        for _attr in attributes:
            # load all non tf attributes
            setattr(self,_attr,attributes[_attr])

        self._load_nonbayes(prefix)

    def _init_MLPGaussianRegressor(self):
        combine_args = copy.deepcopy(self.method_args)
        for _attr in ["activation","dtype"]:
            combine_args.update({_attr:getattr(self,_attr)})

        # a class with key,value as attribute name,value
        args = toy_argparse(combine_args)

        if self.method == "nonbayes":
            final = [int(self.y_dim+self.y_dim*(self.y_dim+1)/2)]
            self.session["ensemble"] = [MLPGaussianRegressor(args,\
                [self.D]+list(self.layers)+final, 'model'+str(i)) for i in range(self.Nensemble)]
            

        elif self.method == "standard_ensemble":
            final = [int(self.y_dim)]
            self.session["ensemble"] = [NoLearnedCovariance(args,\
                [self.D]+list(self.layers)+final, 'model'+str(i)) for i in range(self.Nensemble)]
        
        elif self.method == "single_target_gaussian":
            final = [2]
            self.session["ensemble"] = [SingleTargetGaussianRegressor(args,\
                [self.D]+list(self.layers)+final,'model'+str(i)) for i in range(self.Nensemble)]      

    def _fit_nonbayes(self):
        self.session = {"tf_session":None,"saver":None,"ensemble":None}

        # define tf variables
        self._init_MLPGaussianRegressor()

        self.session["tf_session"] = tf.Session()
        self.session["tf_session"].run(tf.global_variables_initializer())

        # don't want momentum/history of weights from optimization
        self.session["saver"] = tf.train.Saver([_v for _v in tf.global_variables() if "RMSProp" not in _v.name])

        for model in self.session["ensemble"]:
            self.session["tf_session"].run(tf.assign(model.output_mean,self.train_data.target_mean))
            if (self.method in ["nonbayes","single_target_gaussian"]):
                self.session["tf_session"].run(tf.assign(model.output_std,self.train_data.target_std))


        # keep value of minibatch loss so convergence can be checked at end
        self.loss = [[] for ii in range(self.Nensemble)]
        self.rmse = [[] for ii in range(self.Nensemble)]
        self.rmse_val = [[] for ii in range(self.Nensemble)]

        maxiter_per_minibatch = int(1.0/self.batch_size)#10
        num_minibatch = max([1,int(self.maxiter/maxiter_per_minibatch)])

        for model_idx,model in enumerate(self.session["ensemble"]):
            print("Training Network {} of {}".format(model_idx+1,self.Nensemble))
            cntr = 0
            for batch_itr in range(num_minibatch):
                # can train on distinct mini batches for each ensemble
                # x,y = self.train_data.next_batch()
                # feed = {model.input_data: x, model.target_data: y}
                x_batches,y_batches = self.train_data.next_batch_set()
                if (np.mod(batch_itr,10)==0):
                    print("epoch: {} of {}".format(batch_itr,num_minibatch))

                for minibatch_iter in range(maxiter_per_minibatch):
                    feed = {model.input_data: x_batches[minibatch_iter], model.target_data: y_batches[minibatch_iter]}

                    _,loss = self.session["tf_session"].run([model.train_op,model.loss_value], feed)


                    if np.mod(cntr,10)==0:
                        self.loss[model_idx].append(loss)
                        self.rmse[model_idx].append(np.sqrt(mse(self.predict_individual_values(self.train_data.xs[::10],model_idx)[0],self.train_data.ys[::10],multioutput='raw_values')))
                        self.rmse_val[model_idx].append(np.sqrt(mse(self.predict_individual_values(self.train_data.xs_val,model_idx)[0],self.train_data.ys_val,multioutput='raw_values')))

                    cntr += 1
                    if np.mod(cntr,100)==0:
                        # decrease learning rate
                        self.session["tf_session"].run(tf.assign(model.lr,\
                                self.method_args["learning_rate"]*(self.method_args["decay_rate"]**(cntr/100))))


        # for easy slicing upon analysis
        self.loss = np.asarray(self.loss)
        self.rmse = np.asarray(self.rmse)
        self.rmse_val = np.asarray(self.rmse_val)

        # pass in standardized data
        pred_mean,_ = self._predict_nonbayes(self.train_data.xs_standardized)
        val_xs = self.train_data.get_xs_standardized(self.train_data.xs_val)
        pred_mean_val,_ = self._predict_nonbayes(val_xs)

        rmse = np.sqrt(mse(self.train_data.ys,pred_mean,multioutput='raw_values'))
        rmse_val = np.sqrt(mse(self.train_data.ys_val,pred_mean_val,multioutput='raw_values'))
        return rmse, rmse_val

    def _predict_nonbayes(self,xs):
        num_values = xs.shape[0]
        def batch_cov(x,swap_axes=True):
            #batch dimension is 1 (starting from 0)
            #change that to 0 to start
            if (swap_axes):
                x = np.swapaxes(x,1,0)
            N = x.shape[1]
            if (N==1):
                return np.zeros((x.shape[0],x.shape[-1],x.shape[-1]))
            m = x - x.sum(1,keepdims=True)/N
            batch_cov = np.einsum('nji,njk->nik',m,m)/(N-1)
            return batch_cov

        if self.method in ["single_target_gaussian","nonbayes"]:
            means = np.zeros((self.Nensemble,num_values,self.train_data.y_dim))
            vars = np.zeros((self.Nensemble,num_values,self.train_data.y_dim,self.train_data.y_dim))
            cntr = 0

            for ii,model in enumerate(self.session["ensemble"]):
                
                feed = {model.input_data: xs}
                mean,var = self.session["tf_session"].run([model.mean,model.var],feed)
                # if (self.method == "single_target_gaussian"):
                #     var = var.reshape(-1,1,1)
                #     mean = mean.reshape(-1,1,1)
                vars[ii,:,:,:] = var#np.linalg.norm(np.linalg.eigvals(var),axis=1)
                means[ii,:,:] = mean
                #var = np.linalg.eigvals(val)
                    #var = np.array([var[:,0,0],var[:,1,1]]).T
                # en_mean += mean
                # en_var += var + np.sqrt(np.sum(np.square(mean)))

                # count number of samples drawn
                cntr += 1

            en_mean = np.mean(means,axis=0)#dimensions of num_values, y_dim
            en_var = np.mean(vars,axis=0) - batch_cov(means-en_mean)#np.sqrt(np.sum(np.square(en_mean)))#en_mean**2
            en_var = np.linalg.norm(np.linalg.eigvals(en_var),axis=1)

        elif self.method == "standard_ensemble":
            means = np.zeros((self.Nensemble,num_values,self.train_data.y_dim))
            cntr = 0

            for ii,model in enumerate(self.session["ensemble"]):

                feed = {model.input_data: xs}
                mean = self.session["tf_session"].run([model.mean],feed)
                means[ii,:,:] = mean[0]
                cntr += 1

            en_mean = np.mean(means,axis=0)#dimensions of num_values, y_dim
            en_var = batch_cov(means-en_mean)#np.sqrt(np.sum(np.square(en_mean)))#en_mean**2
            en_var = np.linalg.norm(np.linalg.eigvals(en_var),axis=1)

        else: raise NotImplementedError

        return en_mean,en_var

    def _save_nonbayes(self,prefix):
        attributes = {}
        #remove actual data before saving
        self.train_data.xs = None
        self.train_data.ys = None
        self.train_data.xs_standardized = None

        for _attr in [_a for _a in self.__dict__ if _a not in ["session"]]:
            attributes.update({_attr:getattr(self,_attr)})
        with open("./{}/{}.pckl".format(prefix,prefix),"wb") as f:
            pickle.dump(attributes,f)
        f.close()
        self.session["saver"].save(self.session["tf_session"],"./{}/{}".format(prefix,prefix))

        # save tf variables to numpy arrays for use with sklearn
        if (self.method_args["opt_method"]=="rmsprop"):
            self._save_tf_to_np(prefix)

    def _save_tf_to_np(self,prefix):
        # size of ensemble
        Nens = len([_v for _v in tf.global_variables() if "MLP/weights_0:0" in _v.name and "RMSProp" not in _v.name])

        num_weight_layers = len([_v for _v in tf.global_variables() if "model0MLP/weights_" in _v.name \
                and "RMSProp" not in _v.name])

        num_nodes = np.asarray([int(_v.shape[0]) for _v in tf.global_variables() if "model0MLP/weights_" in _v.name\
                and "RMSProp" not in _v.name])

        order = np.asarray([int(_v.name.split(':')[0].split("weights_")[-1]) for _v in tf.global_variables() \
            if "model0MLP/weights" in _v.name and "RMSProp" not in _v.name])

        idx = np.argsort(order)

        layer_units = np.asarray(list(num_nodes[idx]) + [2])

        hidden_layer_sizes = num_nodes[:-1]

        weights = [[None for jj in range(num_weight_layers)] for ii in range(Nens)]
        biases =  [[None for jj in range(num_weight_layers)] for ii in range(Nens)]
        preconditioning = [{"mean":None,"std":None} for ii in range(Nens)]
        for ii in range(Nens):
            for _var in preconditioning[ii]:
                val = self.session["tf_session"].run([_v for _v in tf.global_variables() if \
                        "model{}target_stats/{}".format(ii,_var) in _v.name])[0]
                preconditioning[ii][_var] = val

            for ww in range(num_weight_layers):
                weights[ii][ww] = self.session["tf_session"].run([_v for _v in tf.global_variables() \
                        if "model{}MLP/weights_{}".format(ii,ww) in _v.name and "RMSProp" not in _v.name])[0]
                biases[ii][ww]  = self.session["tf_session"].run([_v for _v in tf.global_variables() \
                        if "model{}MLP/biases_{}".format(ii,ww) in _v.name and "RMSProp" not in _v.name])[0]

        data = {"preconditioning":preconditioning,"activation":self.activation,"hidden_layer_sizes":hidden_layer_sizes,\
            "layer_units":layer_units,"weights":weights,"biases":biases,"Nensemble":Nens}

        with open("{}/netparams-{}.pckl".format(prefix,prefix),"wb") as f:
            pickle.dump(data,f)
        f.close()

    def _load_nonbayes(self,prefix):
        self.session = {"tf_session":None,"ensemble":None,"saver":None}
        self._init_MLPGaussianRegressor()
        self.session["tf_session"] = tf.Session()
        self.session["tf_session"].run(tf.global_variables_initializer())
        #self.session["saver"] = tf.train.Saver(tf.global_variables())
        self.session["saver"] = tf.train.Saver([_v for _v in tf.global_variables() if "RMSProp" not in _v.name])
        self.session["saver"].restore(self.session["tf_session"],"{}/{}".format(prefix,prefix))
    
    def Close_session(self):
        if not self.session["tf_session"]._closed:
            self.session["tf_session"].close
            tf.reset_default_graph()



class GeneralError(Exception):
    pass
