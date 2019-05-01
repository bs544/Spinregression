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
            return tf.add( tf.nn.softplus(x) , tf.fill(dims=tf.shape(x),value=tf.cast(1e-8,dtype=x.dtype)) )
        def normalize_transform(x,K):
            return tf.divide( x , tf.reshape(tf.tile(tf.expand_dims(tf.reduce_sum(x,axis=1), -1),  [1, K]), tf.shape(x)) )
        def mdn_loss(target_values,mean_values,var_values,K):
            # mu_nk - t_n
            exp_term = mean_values - tf.reshape(tf.tile(target_values,  [1, K]), tf.shape(mean_values))

            # -0.5*(mu_nk - t_n)**2 / var_nk
            exp_term = -0.5*tf.divide(tf.square(exp_term),var_values)

            # sum_k pi_nk e^(-0.5 (mu_nk-t_n)**2 / var_nk) * var_nk**-0.5
            ln_term = tf.reduce_sum(tf.multiply(mix_values , tf.divide( tf.exp(exp_term) , tf.sqrt(var_values) ) ) , axis=1)

            # reduce affect of numerical noise
            ln_term += tf.fill(dims=tf.shape(ln_term),value=tf.cast(1e-8,dtype=x.dtype))

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


        #self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        # set method optimizer
        self.optimizer = self.set_optimizer(args)

        #self.train_op = optimizer.apply_gradients(zip(self.clipped_gradients, tvars))
        self.train_op = self.optimizer.apply_gradients(zip(self.gradients, tvars))

    def set_optimizer(self,args):
        if args.opt_method == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif args.opt_method == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif args.opt_method == "gradientdescent":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        else: raise NotImplementedError
        return optimizer

class MLPGaussianRegressor():

    def __init__(self, args, sizes, model_scope):
        def get_tri_diag(n):
            diag_eles = np.zeros(n).astype(int)
            for i in range(n):
                if (np.mod(i,2)==0):
                    if(i==0):
                        diag_eles[i] = 0
                    else:
                        diag_eles[i] = diag_eles[i-1] + 1
                else:
                    diag_eles[i] = diag_eles[i-1] + n
            return diag_eles
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

        target_size = int(-1.5+0.5*np.sqrt(9+8*sizes[-1]))
        o_mean = np.zeros(target_size)
        o_std = 0.1*np.ones((target_size,target_size))

        self.delta = tf.constant(1e-6,dtype=dtype)
        if (False):
            mean = 0.01
            std=0.1
            diag_eles = get_tri_diag(target_size)
            diag_eles = [i+target_size for i in diag_eles]
            idx = np.linspace(0,sizes[-1]-1,sizes[-1]).astype(int)
            condition = [i in diag_eles for i in idx]

        # x,y placeholders
        self.input_data  = tf.placeholder(dtype, [None, sizes[0]])
        self.target_data = tf.placeholder(dtype, [None, target_size])

        with tf.variable_scope(model_scope+'learning_rate'):
            self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate',dtype=dtype)

        with tf.variable_scope(model_scope+'target_stats'):

            self.output_mean = tf.Variable(o_mean, trainable=False, dtype=dtype,name="mean")
            self.output_std = tf.Variable(o_std, trainable=False, dtype=dtype,name="std")
            self.output_cov = np.dot(self.output_std,self.output_std)

        self.weights = []
        self.biases = []

        with tf.variable_scope(model_scope+'MLP'):
            for i in range(1, len(sizes)):
                self.weights.append(tf.Variable(tf.random_normal([sizes[i-1], sizes[i]], stddev=0.1,dtype=dtype), \
                        name='weights_'+str(i-1),dtype=dtype))
                self.biases.append(tf.Variable(tf.random_normal([sizes[i]], stddev=0.1,dtype=dtype), \
                        name='biases_'+str(i-1),dtype=dtype))
            if (False):
                #here follows a bunch of different ways of initialising the biases so that the initial covariances weren't so low
                #It didn't work, but it seemed like a waste to delete them.

                target_biases = tf.Variable(tf.random_normal([target_size], stddev=0.1,dtype=dtype),name='biases_'+str(len(sizes)-1),dtype=dtype)
                cov_biases = tf.Variable(tf.random_normal([sizes[-1]-target_size],stddev=std,mean=mean,dtype=dtype),name='biases_'+str(len(sizes)-1),dtype=dtype)
                final_bias = tf.Variable(tf.concat([target_biases,cov_biases],0),name='biases'+str(len(sizes)-1),dtype=dtype)

            if (False):

                diag_biases = np.random.normal(loc=mean,scale=std, size=sizes[-1])
                other_biases = np.random.normal(scale=0.1,size=sizes[-1])
                final_bias = tf.Variable(np.where(condition,diag_biases,other_biases),name='biases'+str(len(sizes)-1),dtype=dtype)

            if (False):

                diag_biases = tf.Variable(tf.random_normal([sizes[-1]],stddev=std,mean=mean,dtype=dtype),name='diag_biases'+str(len(sizes)-1),dtype=dtype)
                other_biases = tf.Variable(tf.random_normal([sizes[-1]],stddev=0.1,dtype=dtype),name='other_biases'+str(len(sizes)-1),dtype=dtype)
                final_bias = tf.where(condition,diag_biases,other_biases)

            if (False):

                final_bias = tf.Variable(tf.random_normal([sizes[-1]], stddev=0.2,dtype=dtype), name='biases_'+str(len(sizes)-1),dtype=dtype)
                self.weights.append(tf.Variable(tf.random_normal([sizes[-2], sizes[-1]], stddev=0.1,dtype=dtype), name='weights_'+str(len(sizes)-1),dtype=dtype))
                self.biases.append(final_bias)


        x = self.input_data
        for i in range(0, len(sizes)-2):
            x = activation_func(tf.add(tf.matmul(x, self.weights[i]), self.biases[i]))

        self.output = tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1])

        self.mean, self.flat_mat = tf.split(self.output, [target_size,sizes[-1]-target_size], axis=1)

        self.ls = tf.contrib.distributions.fill_triangular(self.flat_mat)

        self.raw_var = tf.matmul(self.ls,self.ls,transpose_b=True)

        #output transform mean
        self.mean = tf.add(tf.matmul(self.mean,self.output_std),self.output_mean)

        self.var = tf.einsum('nij,ij->nij',self.raw_var,self.output_cov)

        # total loss (no regulariation) , sum of log (iid. heteroscedastic errors)
        def gaussian_nll(mean_values, var_values, y):
            eye = tf.eye(num_rows=tf.shape(var_values)[1],batch_shape=[tf.shape(var_values)[0]],dtype=dtype)
            eye = tf.einsum('nij,ij->nij',tf.scalar_mul(self.delta,eye),self.output_cov)

            var_w_eye = tf.add(var_values,eye)
            var_w_eye = tf.einsum('nij,ij->nij',var_w_eye,self.output_cov)

            y_diff = tf.subtract(y, mean_values)
            det = tf.linalg.det(var_values)
            inv = tf.linalg.inv(var_w_eye)
            tmp1 = tf.reduce_mean(tf.log(det))
            tmp2 = tf.reduce_mean(tf.einsum('ij,ijk,ik->i',y_diff,inv,y_diff))
            return tmp1 + tmp2 #+ tmp3

        # objective function (conditional heterscedastic error dist)
        self.loss_value = gaussian_nll(self.mean, self.var, self.target_data)

        tvars = tf.trainable_variables()

        if adversarial_training:
            # need grad_x loss to generate adversarial examples
            self.nll_gradients = tf.gradients(args.alpha * self.loss_value, self.input_data)[0]

            self.adversarial_input_data = tf.add(self.input_data, args.epsilon * tf.sign(self.nll_gradients))

            x_at = self.adversarial_input_data
            for i in range(0, len(sizes)-2):
                x_at = activation_func(tf.add(tf.matmul(x_at, self.weights[i]), self.biases[i]))

            output_at = tf.add(tf.matmul(x_at, self.weights[-1]), self.biases[-1])

            raw_var_e_at, mean_e_at, raw_var_es_at, mean_s_at, raw_var_s_at = tf.split(output_at, [1, 1, 1, 1, 1], axis=1)

            mean_at = tf.concat([mean_e_at,mean_s_at],axis=1)
            raw_var_at = tf.stack([tf.concat([raw_var_e_at,raw_var_es_at],axis=1),tf.concat([raw_var_es_at,raw_var_s_at],axis=1)],axis=1)

            # Output transform
            mean_at = tf.einsum('ij,jk->ik',mean_at,self.output_std)+self.output_mean
            var_at = (tf.nn.softplus(raw_var_at) + 1e-6) * (self.output_std**2)
            #var_at = (tf.log(1 + tf.exp(raw_var_at)) + 1e-6) * (self.output_std**2)

            self.nll_at = gaussian_nll(mean_at, var_at, self.target_data)


            self.gradients = tf.gradients(args.alpha * self.loss_value + (1 - args.alpha) * self.nll_at, tvars)
        else:
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

        if args.opt_method == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif args.opt_method == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        self.train_op = optimizer.apply_gradients(zip(self.clipped_gradients, tvars))
