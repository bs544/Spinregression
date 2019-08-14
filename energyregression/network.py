import numpy as np
import tensorflow as tf
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_AFFINITY'] = 'noverbose'


#network for proof of concept energy regression
#add different element types and force learning later
#also add support for different numbers of atoms in the cell later

class Cell_network():
    def __init__(self,arg_dict,layers,name):
        self.arg_dict= arg_dict
        self.layers = layers
        self.name = name
        self.setup()
    
    def loss_fn(self,E_i,target):
        pred = tf.reduce_sum(E_i,axis=1)
        diff = tf.subtract(pred,target)
        loss = tf.sqrt(tf.reduce_mean(tf.square(diff)))
        return loss
    
    def setup_network(self):

        if (self.arg_dict['activation'] == 'logistic'):
            activation = tf.nn.sigmoid
        elif (self.arg_dict['activation'] == 'tanh'):
            activation = tf.nn.tanh
        elif (self.arg_dict['activation'] == 'relu'):
            activation = tf.nn.relu
        
        dtype = self.arg_dict['dtype']
        
        self.lr = tf.Variable(self.arg_dict['lr'],trainable=False,name='e_learning_rate',dtype=dtype)
        self.input_data = tf.placeholder(dtype,[None,None,self.layers[0]])

        self.weights = []
        self.biases = []
        
        #set up network weights
        with tf.variable_scope(self.name+'energy_net_params'):
            for i in range(1, len(self.layers)):
                self.weights.append(tf.Variable(tf.random_normal([self.layers[i-1], self.layers[i]], stddev=0.1,dtype=dtype), \
                        name='weights_'+str(i-1),dtype=dtype))
                self.biases.append(tf.Variable(tf.random_normal([self.layers[i]], stddev=0.1,dtype=dtype), \
                        name='biases_'+str(i-1),dtype=dtype))
        
        x = self.input_data
        for i in range(0,len(self.layers)-2):
            #this will allow batches of cells and batches of atoms in each cell, likely requires constant number of atoms in each cell, so watch out for that
            x = activation(tf.add(tf.einsum('nai,ij->naj',x,self.weights[i]),self.biases[i]))#tf.nn.xw_plus_b(x,self.weights[i],self.biases[i]))
        
        
        
        self.E_i = tf.add(tf.einsum('nai,ij->naj',x,self.weights[-1]),self.biases[-1])
    
    def set_saver(self):
        self.saver = tf.train.Saver([_v for _v in tf.global_variables() if "RMSProp" not in _v.name])
    
    def set_optimiser(self,method):
        if (method == 'rmsprop'):
            optimiser = tf.train.RMSPropOptimizer(self.lr)
        elif (method == 'adam'):
            optimiser = tf.train.AdamOptimizer(learning_rate=self.lr)
        else: raise NotImplementedError
        return optimiser
        
    def setup(self):
        
        self.setup_network()

        self.target = tf.placeholder(self.arg_dict['dtype'],[None,1])

        self.loss = self.loss_fn(self.E_i,self.target)
        tvars = tf.trainable_variables()
        self.gradients = tf.gradients(self.loss,tvars)
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients,self.arg_dict['grad_clip'])

        self.optimiser = self.set_optimiser(self.arg_dict['opt_method'])

        self.train_op = self.optimiser.apply_gradients(zip(self.clipped_gradients,tvars))



        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.set_saver()
    
    def train_step(self,x,y):
        """
        Parameters:
            x: (array) shape (Ncells,NatomsPerCell,fplen) input data
            y: (array) shape (Ncells,1) target data (energy)
        Actions:
            runs a training step for the neural network energy regression
        """
        feed = {self.input_data:x,self.target:y}
        _,loss = self.session.run([self.train_op,self.loss],feed)
        return loss
    
    def predict(self,x):
        """
        Parameters:
            x: (array) shape (Ncells,NatomsPerCell,fplen) input data
        Returns:
            y: (array) shape (Ncells,1) predicted energy
        """
        feed = {self.input_data:x}
        E_i = self.session.run([self.E_i],feed)[0]
        return np.sum(E_i,axis=1)
    
    def update_lr(self,lr):
        """
        Parameters:
            lr: (float) new learning rate
        Actions:
            assigns the learning rate to be used in the next training step
        """
        self.session.run(tf.assign(self.lr,lr))

    def save_network(self,name):


        attributes = {}
        for _attr in [_a for _a in self.__dict__ if _a in ["arg_dict","layers","name"]]:
            attributes.update({_attr:getattr(self,_attr)})
        with open("./{}/net_{}.pckl".format(name,name),"wb") as f:
            pickle.dump(attributes,f)
        self.saver.save(self.session,"./{}/{}".format(name,name))  

    def load_network(self,name):

        with open('{}/net_{}.pckl'.format(name,name),'rb') as f:
            attributes = pickle.load(f)
        for _attr in attributes:
            # load all non tf attributes
            setattr(self,_attr,attributes[_attr])

        self.saver.restore(self.session,"{}/{}".format(name,name))
    
