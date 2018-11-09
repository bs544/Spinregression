import os
import numpy as np
from sklearn.neural_network import MLPRegressor as mlp
from sklearn.utils import check_random_state
import pickle

class prediction_MLPRegressor():
    """ 
    Class for making predictions only, not for training. Removes overhead of
    tf graphs
    """

    def __init__(self,fname):
        """
        instantiate ensemble of neural nets using sklearn
        """

        if os.path.isdir(fname):
            open_name = "{}/netparams-{}.pckl".format(fname,fname)
        else:
            open_name = fname
        
        with open(open_name,"rb") as f:
            data = pickle.load(f)
        f.close()


        ensemble = [mlp(hidden_layer_sizes=data["hidden_layer_sizes"],activation=data["activation"]) for \
                ii in range(data["Nensemble"])]

        for model in range(data["Nensemble"]):            
            ensemble[model]._random_state = check_random_state(ensemble[model].random_state)
            ensemble[model]._initialize(y=np.zeros((1,2)), layer_units=data["layer_units"])

            for ii in range(len(data["layer_units"])-1):
                ensemble[model].coefs_[ii] = data["weights"][model][ii] 
                ensemble[model].intercepts_[ii] = data["biases"][model][ii]
        
        self.ensemble = ensemble
        self.Nens = data["Nensemble"]
        self.Xdim = data["layer_units"][0]
        self.preconditioning = data["preconditioning"]

    def predict(self,X):
        if X.shape[1]!=self.Xdim:
            raise GeneralError("Dimension of feature dimension for X inconsistent")

        ens_mean,ens_vari = [],[]
        for model in range(self.Nens):
            (mean,vari) = np.array_split(self.ensemble[model].predict(X),2,axis=1)

            # precondition mean and variance
            mean = mean*self.preconditioning[model]["std"] + self.preconditioning[model]["mean"]
            vari = (np.log(np.exp(vari) + 1) + 1e-6 ) * self.preconditioning[model]["std"]**2
            
            ens_mean.append(mean)
            ens_vari.append(vari)
        
        mean_out = np.average(np.asarray(ens_mean),axis=0)
        vari_out = np.average(np.square(ens_mean)+ens_vari , axis=0) - np.square(mean_out)
        
        return mean_out.flatten(),np.sqrt(vari_out).flatten()
