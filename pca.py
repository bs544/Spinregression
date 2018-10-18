import numpy as np
from features.util import format_data

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

class GeneralError(Exception):
    pass

