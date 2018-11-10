import numpy as np
import pickle
import os
from features.util import format_data
from features.bispectrum import local_features,global_features

class PCA():
    def __init__(self,explained_variance=0.99):
        self._set_explained_variance(explained_variance)

    def clear_unecessary_data(self):
        """
        clear uncessary data when making predictions only
        """
        try:
            del self.formatted_data.xs
            del self.formatted_data.xs_standardized
        except AttrubuteError: pass

    def _set_explained_variance(self,explained_variance):
        """
        set explained variance to (0,1]
        """
        if not isinstance(explained_variance,(float,np.float)) or explained_variance<1e-8 or explained_variance>1.0: 
            raise GeneralError("Explained variance must be (0,1]")
        
        self.explained_variance = explained_variance

    def fit(self,X):
        """
        perform linear PCA reduction
        """

        # set mean to zero and standard deviation to 1 where applicable
        self.formatted_data = format_data(X=X)

        # covariance matrix
        C = np.cov(self.formatted_data.xs_standardized.T)

        # diagonalize covariance matrix
        eigval,eigvec = np.linalg.eig(C)
        eigvec = eigvec.T

        # sort in descending eigenvalue order
        idx = np.argsort(eigval)[::-1]

        # covariance matrices are positive semidefinite -> eigenvalues/vecotors are real
        eigval = np.asarray(np.real(eigval[idx]),dtype=np.float64) 
        eigvec = np.asarray(np.real(eigvec[idx]),dtype=np.float64)

        try:
            # identify which eigenvalue completes necessary explained variance
            idx = np.where(np.cumsum(eigval) >= self.explained_variance*np.sum(eigval))[0][0]
        except IndexError:
            idx = eigval.shape[0]

        # which eigvals to use
        idx_to_reduce = np.arange(idx+1)

        self.W = eigvec[0]
        for _id in idx_to_reduce[1:]:
            self.W = np.vstack((self.W,eigvec[_id]))

        # return reduced x
        return self.predict(X)

    def predict(self,X):
        if len(X.shape)==1:
            X = np.reshape(X,(1,-1))

        # mean = 0, std = 1 where applicable
        xnorm = self.formatted_data.get_xs_standardized(X)
        return np.dot(xnorm,self.W.T)


class generate_data():
    def __init__(self,train_frac=1.0,nmax=6,lmax=6,rcut=6.0,local_form="powerspectrum",global_form="powerspectrum",\
    explained_variance=0.99,parallel=True,load=None):
        """
        type(gip) = parsers.GeneralInputParser
        """
        # attributes necessary to make predictions
        self.attributes = ["nmax","lmax","rcut","local_form","global_form","local_pca","global_pca"]

        if load is None:
            self._set_train_frac(train_frac)
            self._set_nmax(nmax)
            self._set_lmax(lmax)
            self._set_rcut(rcut)
            self._set_local_form(local_form)
            self._set_global_form(global_form)
            self._set_explained_variance(explained_variance)
            self._set_parallel(parallel)
        else:
            if not os.path.exists(load):
                raise GeneralError("generate_data file {} not found".format(load))                

            # load attributes from disk
            self.load(load)

            # need a few extra attributes
            self._set_parallel(parallel)

    def _set_train_frac(self,train_frac):
        self.train_frac = train_frac
    def _set_nmax(self,nmax):
        self.nmax = nmax
    def _set_lmax(self,lmax):
        self.lmax = lmax
    def _set_rcut(self,rcut):
        self.rcut = rcut
    def _set_local_form(self,local_form):
        self.local_form = local_form
    def _set_global_form(self,global_form):
        self.global_form = global_form
    def _set_explained_variance(self,explained_variance):
        self.explained_variance = explained_variance
    def _set_parallel(self,parallel):
        self.parallel = parallel

    def save(self,fname):
        """
        save everything needed to run a prediction from disk. If pckl in suffix
        of fname, write fname exactly. Otherwise, write fname/pca-fname.pckl to
        store in single model directory shared between pca and regressor 
        """
        # don't need to store training data
        for _attr in ["local_pca","global_pca"]:
            if getattr(self,_attr) is not None:
                getattr(self,_attr).clear_unecessary_data()

        saveme = {}
        for _attr in self.attributes:
            saveme.update({_attr:getattr(self,_attr)})
        
        if fname.split('.')[-1]=="pckl":
            save_file = fname
        else:
            save_file = "{}/pca-{}.pckl".format(fname,fname)
       
        if not os.path.exists(fname):
            os.mkdir(fname)

        with open(save_file,"wb") as f:
            pickle.dump(saveme,f)
        f.close()            

    def load(self,fname):
        """
        load everything necessary to make predictions. Can give explicit file
        or directory with presumed format for pca pckl file:

        when fname = dir, open dir/pca-dir.pckl
        """
        if os.path.isdir(fname):
            open_file = "{}/pca-{}.pckl".format(fname,fname)
        else:
            open_file = fname

        with open(open_file,"rb") as f:
            data = pickle.load(f)
        f.close()  

        if not all([True if _attr in data.keys() else False for _attr in self.attributes]):
            raise GeneralError("necessary attribute not found in pickle file : {}".format(data.keys()))

        for _attr in data.keys():
            setattr(self,_attr,data[_attr])

    def fit(self,gip):    
        """
        Do 2 separate PCAs, one for local and one for global x
        """

        X_local,X_global,y,natms = self._calculate_features(gip)
       
        if len(X_global.shape)==1:
            # need to check if only one structure is in training set for global descriptor PCA
            X_global = np.reshape(X_global,(1,-1))

        # local PCA 
        self.local_pca = PCA(explained_variance=self.explained_variance) 
        X_local = self.local_pca.fit(X_local)

        # global PCA
        if self.global_form is None:
            self.global_pca = None
            X = X_local
        else:
            if X_global.shape[0]==1:
                raise GeneralError("Cannot use PCA on global descriptors when only 1 structure is in trainin set")

            self.global_pca = PCA(explained_variance=self.explained_variance)
            X_global = self.global_pca.fit(X_global)

            # expand global contributions to full length
            tmp = np.tile(X_global[0],(natms[0],1))
            for ii in range(1,natms.shape[0]):
                tmp = np.vstack((tmp,np.tile(X_global[ii],(natms[ii],1))))
            X_global = tmp            

            # concacenate local and global contributions
            X = np.hstack((X_local,X_global))

        return X,y

    def _calculate_features(self,gip):
        X_local,X_global,y,natms = None,None,None,np.zeros(len(gip.supercells),dtype=int)
        
        for ii,_conf in enumerate(gip.supercells):
            # choose subset of Ntot grid poitns
            Ntot = _conf["edensity"]["xyz"].shape[0]
            if np.isclose(self.train_frac,1.0):
                idx = np.arange(Ntot)
            else:
                idx = np.random.choice(np.arange(Ntot),size=int(Ntot*self.train_frac),replace=False)

            # store number of grid points used from conf
            natms[ii] = idx.shape[0]

            # if densities are given, keep them in case we want to make comparisons
            if _conf["edensity"] is not None:
                if _conf["edensity"]["density"] is not None:
                    if y is None:
                        y = _conf["edensity"]["density"][idx]
                    else:
                        y = np.hstack((y,_conf["edensity"]["density"][idx]))
            
            localx = local_features(cell=_conf["cell"],atom_pos_uvw=_conf["positions"],\
                    xyz=_conf["edensity"]["xyz"][idx],nmax=self.nmax,\
                    lmax=self.lmax,rcut=self.rcut,parallel=self.parallel,form=self.local_form)
            
            if self.global_form is not None:
                globalx = global_features(cell=_conf["cell"],atom_pos_uvw=_conf["positions"],\
                        nmax=self.nmax,lmax=self.lmax,rcut=self.rcut,form=self.global_form)

            if X_local is None:
                X_local = localx
                if self.global_form is not None:
                    X_global = globalx
            else:
                X_local = np.vstack((X_local,localx))
                if self.global_form is not None:
                    X_global = np.vstack((X_global,globalx))
            
        return X_local,X_global,y,natms

    def predict(self,gip):
        """
        Given a class with attribute supercells, itself a list of a different 
        per-configuration class, output the reduced descriptor
        """

        try:
            # _calculate_features takes data selection from self.train_frac
            tmp = self.train_frac
        except AttributeError: pass            
        self.train_frac = 1.0

        X_local,X_global,y ,natms= self._calculate_features(gip)

        try:
            self.train_frac = tmp
        except NameError: pass

        X_local = self.local_pca.predict(X_local)
        if self.global_form is not None:
            X_global = self.global_pca.predict(X_global)
            
            # expand global contributions to full length
            tmp = np.tile(X_global[0],(natms[0],1))
            for ii in range(1,natms.shape[0]):
                tmp = np.vstack((tmp,np.tile(X_global[ii],(natms[ii],1))))
            X_global = tmp            

            # concacenate local and global contributions
            X = np.hstack((X_local,X_global))
        else:
            X = X_local

        return X,y

class GeneralError(Exception):
    pass

