import tensorflow as tf
from edward.models import Normal
from edward import KLqp
import edward
import numpy as np

class vi_bayes():
    def __init__(self,args,layers):
        """
        first arg of layers is dimension 
        """
        dtype = tf.float64


        def neural_net(X):
            h = tf.tanh(tf.matmul(X,weights[0])+biases[0])
            for ii in range(1,len(weights)-1):
                h = tf.tanh(tf.matmul(h,weights[ii])+biases[ii])
            return tf.reshape(tf.matmul(h,weights[-1])+biases[-1],[-1])

        # number of nodes (dimension) of each layer
        nodes = [[layers[0],layers[1]]]
        for ii in range(1,len(layers)-1):
            nodes.append([layers[ii],layers[ii+1]])
        nodes.append([layers[-1],1])            


        input_data = tf.placeholder(dtype=dtype,shape=[None,layers[0]])
        output_data = tf.placeholder(dtype=dtype,shape=[None,1])

        with tf.name_scope("model"):
            weights = [Normal(loc=tf.zeros(nodes[ii],dtype=dtype),scale=tf.ones(nodes[ii],dtype=dtype),\
                    name="w_{}".format(ii)) for ii in range(len(nodes))]

            biases = [Normal(loc=tf.zeros(nodes[ii][1],dtype=dtype),scale=tf.ones(nodes[ii][1],\
                    dtype=dtype),name="b_{}".format(ii)) for ii in range(len(nodes))]

            self.X = tf.placeholder(dtype,[None,layers[0]],name="X")
            self.y = Normal(loc=neural_net(self.X),scale=0.1*tf.cast(tf.fill([tf.shape(self.X)[0],1],1.0),\
                    dtype=dtype),name="y")

        # variational approximations of weight and bias posterior distributions
        qW = [None for ii in range(len(nodes))]
        qb = [None for ii in range(len(nodes))]

        with tf.variable_scope("posterior"):
            for ii in range(len(nodes)):
                with tf.variable_scope("qW_{}".format(ii)):
                    loc = tf.get_variable("loc",nodes[ii],dtype=dtype)
                    scale = tf.nn.softplus(tf.get_variable("scale",nodes[ii],dtype=dtype))

                    # factored variational dist
                    qW[ii] = Normal(loc=loc,scale=scale)

            for ii in range(len(nodes)):
                with tf.variable_scope("qb_{}".format(ii)):
                    loc = tf.get_variable("loc",[nodes[ii][1]],dtype=dtype)
                    scale = tf.nn.softplus(tf.get_variable("scale",[nodes[ii][1]],dtype=dtype))
                    
                    # factored variational dist
                    qb[ii] = Normal(loc=loc,scale=scale)


        self.qW = qW
        self.qb = qb
        self.weights = weights
        self.biases = biases
    
    def fit(self,X,y):
        pairs = {}
        for ii in range(len(self.weights)):
            pairs.update({self.weights[ii]:self.qW[ii],self.biases[ii]:self.qb[ii]})

        inference = KLqp(pairs,data={self.X:X,self.y:y}) 

    def predict(self,X,Nsample):
        pairs={}
        for ii in range(len(self.weights)):
            # use factored distribution averages for weights and biases
            #pairs.update({self.weights[ii]:self.qW[ii].mean(),self.biases[ii]:self.qb[ii].mean()})
            pairs.update({self.weights[ii]:self.qW[ii],self.biases[ii]:self.qb[ii]})
        pairs.update({"X":X})

        y_post = edward.copy(self.y,dict_swap=pairs)

        with tf.Session() as sess:
            #mean,var = tf.nn.moments(y_post.sample(Nsample),feed_dict={self.X:X})
            out = y_post.sample(Nsample)
            #mean,var = tf.nn.moments(y_post.sample(Nsample),axes=1)
        #print(tf.shape(mean),tf.shape(var))            
        #mean = tf.reshape(mean,[-1,1])
        #var = tf.reshape(var,[-1,1])
        print(np.shape(out))
        return mean,np.std(var)

if __name__ == "__main__":
    inst = vi_bayes(None,[7,10,10])
