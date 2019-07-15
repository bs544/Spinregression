"""
This file is an exact copy from https://github.com/vvanirudh/deep-ensembles-uncertainty.git
Author : Anirudh Vemula

This repository (above) is distributed with a GPL v3 license
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

class NoLearnedCovariance():
    def __init__(self, args, sizes, model_scope):
        # total loss (no regulariation) , sum of log (iid. heteroscedastic errors)
        def rmse_loss(mean_values, y):

            y_diff = tf.subtract(y, mean_values)

            tmp2 = tf.reduce_mean(tf.einsum('ni,ni->n',y_diff,y_diff))

            return tmp2

        # activation func
        if args.activation=="logistic":
            activation_func = tf.nn.sigmoid
        elif args.activation=="relu":
            activation_func = tf.nn.relu
        elif args.activation=="tanh":
            activation_func = tf.nn.tanh

        # 32 and 64 bit support
        dtype=args.dtype

        target_size = sizes[-1]
        o_mean = np.zeros(target_size)
        o_std = 0.1*np.ones((target_size,target_size))

        # x,y placeholders
        self.input_data  = tf.placeholder(dtype, [None, sizes[0]])
        self.target_data = tf.placeholder(dtype, [None, target_size])

        with tf.variable_scope(model_scope+'learning_rate'):
            self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate',dtype=dtype)

        with tf.variable_scope(model_scope+'target_stats'):

            self.output_mean = tf.Variable(o_mean, trainable=False, dtype=dtype,name="mean")
            self.output_std = tf.Variable(o_std, trainable=False, dtype=dtype,name="std")

        self.weights = []
        self.biases = []

        with tf.variable_scope(model_scope+'MLP'):
            for i in range(1, len(sizes)):
                self.weights.append(tf.Variable(tf.random_normal([sizes[i-1], sizes[i]], stddev=0.1,dtype=dtype), \
                        name='weights_'+str(i-1),dtype=dtype))
                self.biases.append(tf.Variable(tf.random_normal([sizes[i]], stddev=0.1,dtype=dtype), \
                        name='biases_'+str(i-1),dtype=dtype))

        x = self.input_data
        for i in range(0, len(sizes)-2):
            # x = activation_func(tf.add(tf.matmul(x, self.weights[i]), self.biases[i]))
            x = activation_func(tf.nn.xw_plus_b(x, self.weights[i], self.biases[i]))

        self.mean = tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1])

        #output transform mean
        eye = tf.eye(num_rows=target_size,dtype=dtype)
        self.mean = tf.add(tf.einsum('ni,ij,ij->nj',self.mean,self.output_std,eye),self.output_mean)

        self.loss_value = rmse_loss(self.mean, self.target_data)

        tvars = tf.trainable_variables()

        self.gradients = tf.gradients(self.loss_value , tvars)

        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        self.optimizer = self.set_optimizer(args)


        self.train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients, tvars))

    def set_optimizer(self,args):
        if args.opt_method == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif args.opt_method == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif args.opt_method == "gradientdescent":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        elif args.opt_method == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)#,rho=1.0)
        else: raise NotImplementedError
        return optimizer


class RMSEdropout():
    def __init__(self, args, sizes, model_scope):
        # total loss (no regulariation) , sum of log (iid. heteroscedastic errors)
        def rmse_loss(mean_values, y):

            y_diff = tf.subtract(y, mean_values)

            tmp2 = tf.reduce_mean(tf.einsum('ni,ni->n',y_diff,y_diff))

            return tmp2

        # activation func
        if args.activation=="logistic":
            activation_func = tf.nn.sigmoid
        elif args.activation=="relu":
            activation_func = tf.nn.relu
        elif args.activation=="tanh":
            activation_func = tf.nn.tanh

        # 32 and 64 bit support
        dtype=args.dtype

        target_size = sizes[-1]
        o_mean = np.zeros(target_size)
        o_std = 0.1*np.ones((target_size,target_size))

        # x,y placeholders
        self.input_data  = tf.placeholder(dtype, [None, sizes[0]])
        self.target_data = tf.placeholder(dtype, [None, target_size])

        with tf.variable_scope(model_scope+'learning_rate'):
            self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate',dtype=dtype)

        with tf.variable_scope(model_scope+'target_stats'):

            self.output_mean = tf.Variable(o_mean, trainable=False, dtype=dtype,name="mean")
            self.output_std = tf.Variable(o_std, trainable=False, dtype=dtype,name="std")
        
        self.dr = tf.placeholder(dtype)

        self.weights = []
        self.biases = []

        with tf.variable_scope(model_scope+'MLP'):
            for i in range(1, len(sizes)):
                self.weights.append(tf.Variable(tf.random_normal([sizes[i-1], sizes[i]], stddev=0.1,dtype=dtype), \
                        name='weights_'+str(i-1),dtype=dtype))
                self.biases.append(tf.Variable(tf.random_normal([sizes[i]], stddev=0.1,dtype=dtype), \
                        name='biases_'+str(i-1),dtype=dtype))

        x = self.input_data
        for i in range(0, len(sizes)-2):
            x = activation_func(tf.nn.xw_plus_b(x, self.weights[i], self.biases[i]))
            x = tf.nn.dropout(x, self.dr, noise_shape=[1, sizes[i+1]], name='dropout_layer'+str(i))

        self.mean = tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1])

        #output transform mean
        eye = tf.eye(num_rows=target_size,dtype=dtype)
        self.mean = tf.add(tf.einsum('ni,ij,ij->nj',self.mean,self.output_std,eye),self.output_mean)

        self.loss_value = rmse_loss(self.mean, self.target_data)

        tvars = tf.trainable_variables()

        self.gradients = tf.gradients(self.loss_value , tvars)

        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        self.optimizer = self.set_optimizer(args)


        self.train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients, tvars))

    def set_optimizer(self,args):
        if args.opt_method == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif args.opt_method == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif args.opt_method == "gradientdescent":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        elif args.opt_method == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)#,rho=1.0)
        else: raise NotImplementedError
        return optimizer



class MLPGaussianRegressor():
    def __init__(self, args, sizes, model_scope):
        # total loss (no regulariation) , sum of log (iid. heteroscedastic errors)
        def gaussian_nll(mean_values, var_values, y):

            y_diff = tf.subtract(y, mean_values)
            det = tf.linalg.det(var_values)
            tmp1 = tf.reduce_mean(tf.log(det))

            if (learn_inverse):
                tmp1 = -tmp1
                inv = var_values
            else:
                inv = tf.linalg.inv(var_values)

            tmp2 = tf.reduce_mean(tf.einsum('ni,nij,nj->n',y_diff,inv,y_diff))

            return tmp1 + tmp2

        # activation func
        if args.activation=="logistic":
            activation_func = tf.nn.sigmoid
        elif args.activation=="relu":
            activation_func = tf.nn.relu
        elif args.activation=="tanh":
            activation_func = tf.nn.tanh

        learn_inverse = args.learn_inverse

        # 32 and 64 bit support
        dtype=args.dtype

        target_size = int(-1.5+0.5*np.sqrt(9+8*sizes[-1]))
        o_mean = np.zeros(target_size)
        o_std = 0.1*np.ones((target_size,target_size))

        # x,y placeholders
        self.input_data  = tf.placeholder(dtype, [None, sizes[0]])
        self.target_data = tf.placeholder(dtype, [None, target_size])

        with tf.variable_scope(model_scope+'learning_rate'):
            self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate',dtype=dtype)

        with tf.variable_scope(model_scope+'target_stats'):

            self.output_mean = tf.Variable(o_mean, trainable=False, dtype=dtype,name="mean")
            self.output_std = tf.Variable(o_std, trainable=False, dtype=dtype,name="std")
            self.output_cov = tf.linalg.matmul(self.output_std,self.output_std,transpose_b=True)
            self.output_inv_cov = tf.linalg.inv(self.output_cov)

        self.weights = []
        self.biases = []

        with tf.variable_scope(model_scope+'MLP'):
            for i in range(1, len(sizes)):
                self.weights.append(tf.Variable(tf.random_normal([sizes[i-1], sizes[i]], stddev=0.1,dtype=dtype), \
                        name='weights_'+str(i-1),dtype=dtype))
                self.biases.append(tf.Variable(tf.random_normal([sizes[i]], stddev=0.1,dtype=dtype), \
                        name='biases_'+str(i-1),dtype=dtype))

        x = self.input_data
        for i in range(0, len(sizes)-2):
            x = activation_func(tf.add(tf.matmul(x, self.weights[i]), self.biases[i]))

        self.output = tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1])

        self.mean, self.flat_mat = tf.split(self.output, [target_size,sizes[-1]-target_size], axis=1)

        self.ls = tf.contrib.distributions.fill_triangular(self.flat_mat)

        self.ls = tfp.distributions.matrix_diag_transform(self.ls,transform=tf.nn.softplus)

        self.raw_inv_var = tf.einsum('nik,njk->nij',self.ls,self.ls)

        #output transform mean
        eye = tf.eye(num_rows=target_size,dtype=dtype)
        self.mean = tf.add(tf.einsum('ni,ij,ij->nj',self.mean,self.output_std,eye),self.output_mean)

        if (learn_inverse):
            self.inv_var = tf.einsum('nij,ij->nij',self.raw_inv_var,self.output_inv_cov)
            self.var = tf.linalg.inv(self.inv_var)
            # objective function (conditional heterscedastic error dist)
            self.loss_value = gaussian_nll(self.mean, self.inv_var, self.target_data)

        else:
            self.var = tf.einsum('nij,ij->nij',self.raw_inv_var,self.output_cov)
            self.loss_value = gaussian_nll(self.mean,self.var,self.target_data)

        tvars = tf.trainable_variables()

        self.gradients = tf.gradients(self.loss_value , tvars)

        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        self.optimizer = self.set_optimizer(args)


        self.train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients, tvars))

    def set_optimizer(self,args):
        if args.opt_method == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif args.opt_method == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif args.opt_method == "gradientdescent":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        elif args.opt_method == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)#,rho=1.0)
        else: raise NotImplementedError
        return optimizer


class SingleTargetGaussianRegressor():
    def __init__(self, args, sizes, model_scope):
        # total loss (no regulariation) , sum of log (iid. heteroscedastic errors)
        def gaussian_nll(mean_values, var_values, y):

            y_diff = tf.subtract(y, mean_values)
            
            tmp1 = tf.reduce_mean(tf.log(var_values))

            inv = tf.linalg.inv(var_values)

            tmp2 = tf.reduce_mean(tf.div(tf.square(y_diff),var_values))

            return tmp1 + tmp2

        # activation func
        if args.activation=="logistic":
            activation_func = tf.nn.sigmoid
        elif args.activation=="relu":
            activation_func = tf.nn.relu
        elif args.activation=="tanh":
            activation_func = tf.nn.tanh

        # 32 and 64 bit support
        dtype=args.dtype

        #mean may need to be a vector, and std a matrix, in order to interface with the rest of the code
        o_mean = [0.0]
        o_std = [[0.1]]

        # x,y placeholders
        self.input_data  = tf.placeholder(dtype, [None, sizes[0]])
        self.target_data = tf.placeholder(dtype, [None, 1])

        with tf.variable_scope(model_scope+'learning_rate'):
            self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate',dtype=dtype)

        with tf.variable_scope(model_scope+'target_stats'):

            self.output_mean = tf.Variable(o_mean, trainable=False, dtype=dtype,name="mean")
            self.output_std = tf.Variable(o_std, trainable=False, dtype=dtype,name="std")
            self.output_var = tf.square(self.output_std)

        self.weights = []
        self.biases = []

        with tf.variable_scope(model_scope+'MLP'):
            for i in range(1, len(sizes)):
                self.weights.append(tf.Variable(tf.random_normal([sizes[i-1], sizes[i]], stddev=0.1,dtype=dtype), \
                        name='weights_'+str(i-1),dtype=dtype))
                self.biases.append(tf.Variable(tf.random_normal([sizes[i]], stddev=0.1,dtype=dtype), \
                        name='biases_'+str(i-1),dtype=dtype))

        x = self.input_data
        for i in range(0, len(sizes)-2):
            x = activation_func(tf.add(tf.matmul(x, self.weights[i]), self.biases[i]))

        self.output = tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1])

        self.raw_mean, self.raw_var = tf.split(self.output, [1,1], axis=1)

        #output transform mean
        self.mean = tf.add(self.output_mean[0],tf.multiply(self.raw_mean,self.output_std[0,0]))

        self.in_var = tf.multiply(self.raw_var,self.output_var[0,0])

        self.var = tf.reshape(self.in_var,[-1,1,1])

        #self.var = tf.square(self.std)

        self.loss_value = gaussian_nll(self.mean,self.in_var,self.target_data)

        tvars = tf.trainable_variables()

        self.gradients = tf.gradients(self.loss_value , tvars)

        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        self.optimizer = self.set_optimizer(args)


        self.train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients, tvars))

    def set_optimizer(self,args):
        if args.opt_method == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif args.opt_method == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif args.opt_method == "gradientdescent":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        elif args.opt_method == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)#,rho=1.0)
        else: raise NotImplementedError
        return optimizer
