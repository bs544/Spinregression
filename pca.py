import numpy as np
from features.util import format_data
from features.bispectrum import local_features,global_features

class PCA():
    def __init__(self,explained_variance=0.99):
        self._set_explained_variance(explained_variance)

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
        # mean = 0, std = 1 where applicable
        xnorm = self.formatted_data.get_xs_standardized(X)

        return np.dot(xnorm,self.W.T)


class generate_data():
    def __init__(self,train_frac,nmax,lmax,rcut,local_form="powerspectrum",global_form="powerspectrum",\
    explained_variance=0.99,parallel=True):
        """
        type(gip) = parsers.GeneralInputParser
        """
        self._set_train_frac(train_frac)
        self._set_nmax(nmax)
        self._set_lmax(lmax)
        self._set_rcut(rcut)
        self._set_local_form(local_form)
        self._set_global_form(global_form)
        self._set_explained_variance(explained_variance)
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

    def fit(self,gip):    
        """
        Do 2 separate PCAs, one for local and one for global x
        """

        X_local,X_global,y,natms = self._calculate_features(gip)
        
        # local PCA 
        self.local_pca = PCA(explained_variance=self.explained_variance) 
        X_local = self.local_pca.fit(X_local)

        # global PCA
        if self.global_form is None:
            self.global_pca = None
            X = X_local
        else:
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
        
        for ii,_conf in enumerate(gip):
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

        # _calculate_features takes data selection from self.train_frac
        tmp = self.train_frac
        self.train_frac = 1.0

        X_local,X_global,y ,natms= self._calculate_features(gip)

        self.train_frac = tmp

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

