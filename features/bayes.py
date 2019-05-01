import tensorflow as tf
#from tensorflow_probability.distributions import Normal
#from edward.models import Normal
# from edward import KLqp
# from edward import evaluate
#import edward
import numpy as np

class vi_bayes():
    def __init__(self,args,layers):
        """
        first arg of layers is dimension
        """
        self.dtype = args.dtype


        #def neural_net(X):
        #    h = tf.tanh(tf.matmul(X,self.weights[0])+self.biases[0])
        #    for ii in range(1,len(self.weights)-1):
        #        h = tf.tanh(tf.matmul(h,self.weights[ii])+self.biases[ii])
        #    return tf.reshape(tf.matmul(h,self.weights[-1])+self.biases[-1],[-1])

        # number of nodes (dimension) of each layer
        nodes = [[layers[0],layers[1]]]
        for ii in range(1,len(layers)-1):
            nodes.append([layers[ii],layers[ii+1]])
        nodes.append([layers[-1],1])


        input_data = tf.placeholder(dtype=self.dtype,shape=[None,layers[0]])
        output_data = tf.placeholder(dtype=self.dtype,shape=[None,1])

        with tf.name_scope("model"):
            self.weights = [Normal(loc=tf.zeros(nodes[ii],dtype=self.dtype),scale=tf.ones(nodes[ii],dtype=self.dtype),\
                    name="w_{}".format(ii)) for ii in range(len(nodes))]

            self.biases = [Normal(loc=tf.zeros(nodes[ii][1],dtype=self.dtype),scale=tf.ones(nodes[ii][1],\
                    dtype=self.dtype),name="b_{}".format(ii)) for ii in range(len(nodes))]

            self.X = tf.placeholder(self.dtype,[None,layers[0]],name="X")

            # need to change precision to be RV here with some prior
            self.y = Normal(loc=self.neural_net(self.X),scale=0.1*tf.cast(tf.fill([tf.shape(self.X)[0],1],1.0),\
                    dtype=self.dtype),name="y")

        # variational approximations of weight and bias posterior distributions
        self.qW = [None for ii in range(len(nodes))]
        self.qb = [None for ii in range(len(nodes))]

        with tf.variable_scope("posterior"):
            for ii in range(len(nodes)):
                with tf.variable_scope("qW_{}".format(ii)):
                    loc = tf.get_variable("loc",nodes[ii],dtype=self.dtype)
                    scale = tf.nn.softplus(tf.get_variable("scale",nodes[ii],dtype=self.dtype))

                    # factored variational dist
                    self.qW[ii] = Normal(loc=loc,scale=scale)

            for ii in range(len(nodes)):
                with tf.variable_scope("qb_{}".format(ii)):
                    loc = tf.get_variable("loc",[nodes[ii][1]],dtype=self.dtype)
                    scale = tf.nn.softplus(tf.get_variable("scale",[nodes[ii][1]],dtype=self.dtype))

                    # factored variational dist
                    self.qb[ii] = Normal(loc=loc,scale=scale)


    def neural_net(self,X):
        h = tf.tanh(tf.matmul(X,self.weights[0])+self.biases[0])
        for ii in range(1,len(self.weights)-1):
            h = tf.tanh(tf.matmul(h,self.weights[ii])+self.biases[ii])
        return tf.reshape(tf.matmul(h,self.weights[-1])+self.biases[-1],[-1])

    def fit(self,X,y):
        pairs = {}
        for ii in range(len(self.weights)):
            pairs.update({self.weights[ii]:self.qW[ii],self.biases[ii]:self.qb[ii]})

        inference = KLqp(pairs,data={self.X:X,self.y:y})
        inference.run(logdir="log")

        mse = evaluate("mean_squared_error",data={self.y:y,self.X:X})
        return np.sqrt(mse)

    def predict(self,X,Nsample):
        Nsample = 3

        # sample from posterior
        W_post = [self.qW[ii].sample(Nsample).eval() for ii in range(len(self.weights))]
        b_post = [self.qb[ii].sample(Nsample).eval() for ii in range(len(self.weights))]

        aleatoric_noise = Normal(loc=tf.cast(tf.fill([tf.shape(X)[0],1],0.0),dtype=self.dtype),\
                scale=tf.cast(tf.fill([tf.shape(X)[0],1],0.1),dtype=self.dtype))

        noise_post = aleatoric_noise.sample(Nsample).eval()

        prediction = [None for ii in range(Nsample)]
        for ii in range(Nsample):
            self.weights = [W_post[ww][ii] for ww in range(len(self.weights))]
            self.biases  = [b_post[ww][ii] for ww in range(len(self.weights))]

            self.weights = [self.qW[ww].mean() for ww in range(len(self.weights))]
            self.biases  = [self.qb[ww].mean() for ww in range(len(self.weights))]

            prediction[ii] = tf.reshape(self.neural_net(X),[-1])

        prediction_mean,prediction_var = tf.nn.moments(tf.stack(prediction),axes=[0])

        # convert tensor -> numpy.ndarray
        prediction_mean = prediction_mean.eval()
        prediction_var  = prediction_var.eval()


        return prediction_mean,np.sqrt(prediction_var)
            #if ii==0:
            #    prediction = output
            #else:
            #    prediction = tf.stack([prediction,output],axis=0)
            #print(np.shape(prediction))
            #y = self.neural_net(X) #+noise_post[ii]
            #print(np.shape(y),type(y),type(y[0]))

            #pairs={}
            #for ww in range(len(self.weights)):
            #    # use factored distribution averages for weights and biases
            #    pairs.update({self.weights[ww]:W_post[ww][ii],self.biases[ww]:b_post[ww][ii]})
            #
            #y_post = edward.copy(self.y,dict_swap=pairs)

            #with tf.Session() as sess:
            #    prediction[ii] = sess.run(y_post.sample(1),feed_dict={self.X:X})

            #    print(np.shape(prediction[ii]))

            #with tf.Session() as sess:
            #    #prediction[ii] = sess.run(y_post,feed_dict={"model/X:0":X})
            #    prediction[ii] = y_post.eval({self.X:X})

            #    print(prediction[ii])
        #prediction = tf.cast(prediction,tf.float64)

        #with tf.Session() as sess:
        #    #mean,var = tf.nn.moments(y_post.sample(Nsample),feed_dict={self.X:X})
        #    out = y_post.sample(Nsample).eval()
        #    #mean,var = tf.nn.moments(y_post.sample(Nsample),axes=1)
        ##print(tf.shape(mean),tf.shape(var))
        ##mean = tf.reshape(mean,[-1,1])
        ##var = tf.reshape(var,[-1,1])
        #print(np.shape(out))
        #print(np.mean(out))
        #return mean,np.std(var)

if __name__ == "__main__":
    inst = vi_bayes(None,[7,10,10])
