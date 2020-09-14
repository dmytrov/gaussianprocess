import numpy as np
import numerical.numpytheano as nt
import numerical.numpyext.linalg as ntlinalg
import ml.gptheano.vecgpdm.equations as eq

class GPFunction(object):
    inv_mode = eq.InvMode.cholesky

    @classmethod
    def inv_op(cls, x, ns=nt.NumpyLinalg):
        if cls.inv_mode == eq.InvMode.simple:
            return ns.inv(x)
        elif cls.inv_mode == eq.InvMode.cholesky:
            return ns.choleskyinv(x)

    def __init__(self, kernelobj, fmean=None, Dx=1, Dy=1, ns=nt.NumpyLinalg):
        super(GPFunction, self).__init__()
        self.kernelobj = kernelobj
        self.fmean = fmean
        self.Dx = Dx
        self.Dy = Dy
        self.ns = ns
        if self.fmean is None:
            self.fmean = lambda X: np.zeros(shape=(X.shape[0], self.Dy))
        self.sampledX = np.empty(shape=(0, Dx))  # samples inputs
        self.sampledYzeromean = np.empty(shape=(0, Dy))  # zero-mean samples outputs
        self.K = np.array([[]])
        self.Kinv = np.array([[]])

    def sample_values(self, X):
        """
        Extends the function at X conditioned on stored values.
        Y ~ p(\mu(X), K_{xx} | X_stored, Y_stored)
        """
        mean, covar = self.posterior_predictive(X)
        Y = np.vstack([mean[:, d] + np.random.multivariate_normal(mean=0*mean[:, d], cov=covar) for d in range(self.Dy)]).T
        self.condition_on(X, Y)
        return Y        
    
    def condition_on(self, X, Y):
        """
        Add values to conditioning storage
        """
        #print(self.sampledX.shape, X.shape)
        self.sampledX = np.vstack([self.sampledX, X])
        self.sampledYzeromean = np.vstack([self.sampledYzeromean, Y - self.fmean(X)])
        self.K = self.kernelobj.gram_matrix(self.sampledX, self.sampledX)
        self.Kinv = GPFunction.inv_op(self.K, self.ns)
        
    def posterior_predictive(self, Xstar):
        """
        Computes posterior predictive mean and covariance
        """
        if Xstar.ndim < 2:
            Xstar = Xstar[np.newaxis, :]
        if self.sampledX.shape[0] == 0:
            Ystar_mean = self.fmean(Xstar)
            Ystar_cov = self.kernelobj.gram_matrix(Xstar, Xstar)
        else:
            Kxstarx = self.kernelobj.gram_matrix(Xstar, self.sampledX)
            Ystar_mean = self.fmean(Xstar) + Kxstarx.dot(self.Kinv.dot(self.sampledYzeromean))
            Ystar_cov = self.kernelobj.gram_matrix(Xstar, Xstar) - Kxstarx.dot(self.Kinv.dot(Kxstarx.T))
        return Ystar_mean, Ystar_cov
        
    
    
class GP(object):
    def __init__(self, kernelobj, fmean=None, Dx=1, Dy=1, ns=nt.NumpyLinalg):
        super(GP, self).__init__()
        self.kernelobj = kernelobj
        self.fmean = fmean
        self.Dx = Dx
        self.Dy = Dy
        self.ns = ns
        if self.fmean is None:
            self.fmean = lambda X: np.zeros(shape=(X.shape[0], self.Dy))
        
    def sample_function(self):
        """
        Samples a random function
        """
        return GPFunction(self.kernelobj, self.fmean, self.Dx, self.Dy, ns=self.ns)
        


import matplotlib.pyplot as plt
import matplotlibex as plx
import unittest
import ml.gptheano.vecgpdm.kernels as krn
import numerical.numpytheano.theanopool as tp

class TestRBFKernel(unittest.TestCase):

    def test_GP(self):
        D = 2  # ndims
        ns = tp.NumpyVarPool()
        kernelobj = krn.ARD_RBF_Kernel(ndims=D, kern_width=1.0, suffix="", ns=ns)
        gp = GP(kernelobj, Dx=D)
        gpf = gp.sample_function()
        N = 20
        x = np.linspace(0.0, 10.0, N)[:, np.newaxis]        
        x = np.tile(x, reps=[1, D])# np.hstack([x for i in range()])
        #print(x)
        y = gpf.sample_values(x)
        xstar = x + 0.5
        #xstar = np.array([[0.5]])
        xstar = np.linspace(0.0, 10.0, 10*N)[:, np.newaxis]        
        ystar, _ = gpf.posterior_predictive(xstar)
        print(xstar.shape, ystar.shape)
        plt.plot(xstar, ystar, ".", markersize=1)
        plt.plot(x, y, "*")
        plt.show()







if __name__ == '__main__':
    unittest.main()

    