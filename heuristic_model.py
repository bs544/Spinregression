"""
This file is an exact copy from https://github.com/vvanirudh/deep-ensembles-uncertainty.git
Author : Anirudh Vemula

This repository (above) is distributed with a GPL v3 license
"""

import tensorflow as tf
import numpy as np

class MLPDensityMixtureRegressor():

    def __init__(self, args, sizes, model_scope):
        def softplus_transform(x):
            return tf.add( tf.nn.softplus(x) , tf.fill(dims=tf.shape(x),value=tf.cast(1e-6,dtype=x.dtype)) )
        def normalize_transform(x,K):
            return tf.divide( x , tf.reshape(tf.tile(tf.expand_dims(tf.reduce_sum(x,axis=1), -1),  [1, K]), tf.shape(x)) )
        def mdn_loss(target_values,mean_values,var_values,mix_values,K):
            # mu_nk - t_n
            exp_term = mean_values - tf.reshape(tf.tile(target_values,  [1, K]), tf.shape(mean_values))

            # -0.5*(mu_nk - t_n)**2 / var_nk
            exp_term = -0.5*tf.divide(tf.square(exp_term),var_values)

            # sum_k pi_nk e^(-0.5 (mu_nk-t_n)**2 / var_nk) * var_nk**-0.5
            ln_term = tf.reduce_sum(tf.multiply(mix_values , tf.divide( tf.exp(exp_term) , tf.sqrt(var_values) ) ) , axis=1)

            # mixture density loss
            loss = -tf.reduce_sum(tf.log(ln_term))

            return loss

        # activation func
        if args.activation=="logistic":
            activation_func = tf.nn.sigmoid
        elif args.activation=="relu":
            activation_func = tf.nn.relu
        elif args.activation=="tanh":
            activation_func = tf.nn.tanh
     
        # number of components in mixture 
        K = args.num_components

        # when alpha = 1, adversarial training contribution is 0
        adversarial_training = not np.isclose(args.alpha,1.0)


        # 32 and 64 bit support
        dtype=args.dtype

        # x,y placeholders
        self.input_data  = tf.placeholder(dtype, [None, sizes[0]])
        self.target_data = tf.placeholder(dtype, [None, 1])

        with tf.variable_scope(model_scope+'learning_rate'):
            self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate',dtype=dtype)

        with tf.variable_scope(model_scope+'target_stats'):
            self.output_mean = tf.Variable(0., trainable=False, dtype=dtype)
            self.output_std = tf.Variable(0.1, trainable=False, dtype=dtype)

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

        # size of subtensors to split output into
        split_size = tf.convert_to_tensor([K,K,K])

        # output nodes = [mean_1,..,mean_K,var_1,...,var_K,mixing_coeff_1,...,mixing_coeff_K]
        self.mean, self.raw_var, self.raw_mix_coeff = tf.split(self.output, split_size, axis=1)

        # Output transform - WHY?
        self.mean = tf.multiply(self.mean, tf.fill(dims=tf.shape(self.mean),value=self.output_std) )+ \
                tf.fill(dims=tf.shape(self.mean),value=self.output_mean)
        #self.mean = self.mean
        
        # ensure positivity by taking softplus (with small addition for numerical stability)
        self.var = tf.multiply(softplus_transform(self.raw_var) , \
                tf.fill(dims=tf.shape(self.mean),value=self.output_std**2) )
        #self.var = softplus_transform(self.raw_var) 

        # ensure mixing coefficients are positive, add dx for numerical stability
        self.mix_coeff = softplus_transform(self.raw_mix_coeff) 

        # normalize so sum_k mix_coeff = 1
        self.mix_coeff = normalize_transform(self.mix_coeff,K) 

        # mixture density loss function
        self.loss_value = mdn_loss(self.target_data,self.mean,self.var,self.mix_coeff,K)
       

        tvars = tf.trainable_variables()

        if adversarial_training:
            raise NotImplementedError

            # need grad_x loss to generate adversarial examples
            self.nll_gradients = tf.gradients(args.alpha * self.nll, self.input_data)[0]
            
            self.adversarial_input_data = tf.add(self.input_data, args.epsilon * tf.sign(self.nll_gradients))

            x_at = self.adversarial_input_data
            for i in range(0, len(sizes)-2):
                x_at = activation_func(tf.add(tf.matmul(x_at, self.weights[i]), self.biases[i]))

            output_at = tf.add(tf.matmul(x_at, self.weights[-1]), self.biases[-1])

            mean_at, raw_var_at = tf.split(output_at, [1, 1], axis=1)

            # Output transform
            mean_at = mean_at * self.output_std + self.output_mean
            var_at = (tf.nn.softplus(raw_var_at) + 1e-6) * (self.output_std**2)
            #var_at = (tf.log(1 + tf.exp(raw_var_at)) + 1e-6) * (self.output_std**2)

            self.nll_at = gaussian_nll(mean_at, var_at, self.target_data)


            self.gradients = tf.gradients(args.alpha * self.nll + (1 - args.alpha) * self.nll_at, tvars)
        else:
            self.gradients = tf.gradients(self.loss_value , tvars)
            

        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        optimizer = tf.train.RMSPropOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(self.clipped_gradients, tvars))

class MLPGaussianRegressor():

    def __init__(self, args, sizes, model_scope):
        # activation func
        if args.activation=="logistic":
            activation_func = tf.nn.sigmoid
        elif args.activation=="relu":
            activation_func = tf.nn.relu
        elif args.activation=="tanh":
            activation_func = tf.nn.tanh
      
        # when alpha = 1, adversarial training contribution is 0
        adversarial_training = not np.isclose(args.alpha,1.0)


        # 32 and 64 bit support
        dtype=args.dtype

        # x,y placeholders
        self.input_data  = tf.placeholder(dtype, [None, sizes[0]])
        self.target_data = tf.placeholder(dtype, [None, 1])

        with tf.variable_scope(model_scope+'learning_rate'):
            self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate',dtype=dtype)

        with tf.variable_scope(model_scope+'target_stats'):
            self.output_mean = tf.Variable(0., trainable=False, dtype=dtype)
            self.output_std = tf.Variable(0.1, trainable=False, dtype=dtype)

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

        self.mean, self.raw_var = tf.split(self.output, [1,1], axis=1)

        # Output transform - WHY?
        self.mean = self.mean * self.output_std + self.output_mean
        
        # ensure positivity by taking softplus (with small addition for numerical stability)
        self.var = (tf.nn.softplus(self.raw_var) + 1e-6) * (self.output_std**2)
        #self.var = (tf.log(1 + tf.exp(self.raw_var)) + 1e-6) * (self.output_std**2)

        # total loss (no regulariation) , sum of log (iid. heteroscedastic errors)
        def gaussian_nll(mean_values, var_values, y):
            y_diff = tf.subtract(y, mean_values)
            tmp1 = 0.5*tf.reduce_mean(tf.log(var_values))
            tmp2 = 0.5*tf.reduce_mean(tf.div(tf.square(y_diff), var_values)) 
            #tmp3 = 0.5*tf.log(tf.cast(2*np.pi,dtype=dtype))
            return tmp1 + tmp2 #+ tmp3

        # objective function (conditional heterscedastic error dist)
        self.nll = gaussian_nll(self.mean, self.var, self.target_data)

        tvars = tf.trainable_variables()

        if adversarial_training:
            # need grad_x loss to generate adversarial examples
            self.nll_gradients = tf.gradients(args.alpha * self.nll, self.input_data)[0]
            
            self.adversarial_input_data = tf.add(self.input_data, args.epsilon * tf.sign(self.nll_gradients))

            x_at = self.adversarial_input_data
            for i in range(0, len(sizes)-2):
                x_at = activation_func(tf.add(tf.matmul(x_at, self.weights[i]), self.biases[i]))

            output_at = tf.add(tf.matmul(x_at, self.weights[-1]), self.biases[-1])

            mean_at, raw_var_at = tf.split(output_at, [1, 1], axis=1)

            # Output transform
            mean_at = mean_at * self.output_std + self.output_mean
            var_at = (tf.nn.softplus(raw_var_at) + 1e-6) * (self.output_std**2)
            #var_at = (tf.log(1 + tf.exp(raw_var_at)) + 1e-6) * (self.output_std**2)

            self.nll_at = gaussian_nll(mean_at, var_at, self.target_data)


            self.gradients = tf.gradients(args.alpha * self.nll + (1 - args.alpha) * self.nll_at, tvars)
        else:
            self.gradients = tf.gradients(self.nll , tvars)
            

        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        optimizer = tf.train.RMSPropOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(self.clipped_gradients, tvars))


class MLPDropoutGaussianRegressor():

    def __init__(self, args, sizes, model_scope):
        if args.activation=="logistic":
            activation_func = tf.nn.sigmoid
        elif args.activation=="relu":
            activation_func = tf.nn.reul
        elif args.activation=="tanh":
            activation_func = tf.nn.tanh
       
        dtype=args.dtype

        self.input_data  = tf.placeholder(dtype, [None, sizes[0]])
        self.target_data = tf.placeholder(dtype, [None, sizes[0]])

        with tf.variable_scope(model_scope+'learning_rate'):
            self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate',dtype=dtype)

        with tf.variable_scope(model_scope+'target_stats'):
            self.output_mean = tf.Variable(0., trainable=False, dtype=dtype)
            self.output_std = tf.Variable(0.1, trainable=False, dtype=dtype)

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

        self.output = tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1])

        self.mean, self.raw_var = tf.split(axis=1, num_or_size_splits=[1,1], value=self.output)

        # Output transform
        self.mean = self.mean * self.output_std + self.output_mean
        self.var = (tf.log(1 + tf.exp(self.raw_var)) + 1e-6) * (self.output_std**2)

        def gaussian_nll(mean_values, var_values, y):
            y_diff = tf.subtract(y, mean_values)
            tmp1 = 0.5*tf.reduce_mean(tf.log(var_values))
            tmp2 = 0.5*tf.reduce_mean(tf.div(tf.square(y_diff), var_values))
            tmp3 = 0.5*tf.log(tf.cast(2*(np.pi),dtype=dtype))
            return tmp1 + tmp2 + tmp3 

        self.nll = gaussian_nll(self.mean, self.var, self.target_data)

        self.nll_gradients = tf.gradients(args.alpha * self.nll, self.input_data)[0]
        self.adversarial_input_data = tf.add(self.input_data, args.epsilon * tf.sign(self.nll_gradients))

        x_at = self.adversarial_input_data
        for i in range(0, len(sizes)-2):
            #x_at = tf.nn.relu(tf.nn.xw_plus_b(x_at, self.weights[i], self.biases[i]))
            x_at = activation_func(tf.nn.xw_plus_b(x_at, self.weights[i], self.biases[i]))
            # We need to apply the same dropout mask as before
            # so that we maintain the same model and not change the network
            graph = tf.get_default_graph()
            mask = graph.get_tensor_by_name('dropout_layer'+str(i)+'/Floor:0')
            x_at = tf.multiply(x_at, mask)

        output_at = tf.add(tf.matmul(x_at, self.weights[-1]), self.biases[-1])

        mean_at, raw_var_at = tf.split(axis=1, num_or_size_splits=[1,1], value=output_at)

        # Output transform
        mean_at = mean_at * self.output_std + self.output_mean
        var_at = (tf.log(1 + tf.exp(raw_var_at)) + 1e-6) * (self.output_std**2)

        self.nll_at = gaussian_nll(mean_at, var_at, self.target_data)

        tvars = tf.trainable_variables()


        self.gradients = tf.gradients(args.alpha * self.nll + (1 - args.alpha) * self.nll_at, tvars)

        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        optimizer = tf.train.RMSPropOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(self.clipped_gradients, tvars))
