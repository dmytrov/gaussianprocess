from scipy import optimize
import numerical.theanoext as thx
import numerical.theanoext.parametricfunctionminimization as pfm
import numerical.numpyext.dimreduction as dr
from ml.gptheano.kernels import *


class GPLVMNegLogLik(ParametricFunction):
    def __init__(self, kernel, YVar, XVar=None):
        super(GPLVMNegLogLik, self).__init__()
        if XVar is None:
            XVar = kernel.get_args()[0]
        self.YVar = YVar
        self.XVar = XVar
        self.N, self.Q = self.XVar.val.shape
        self._params = [self.YVar]
        self.kernel = kernel
        self._children = [kernel]
        self.symbolic = -eq.gplvm_loglik(self.Q, self.N, self.kernel.symbolic, self.YVar.symbolic, ns=nt.TheanoLinalg)
        self.function = lambda X: -eq.gplvm_loglik(self.Q, self.N, self.kernel.function(X, X), self.YVar.val, ns=nt.NumpyLinalg)

class GPLVM(object):
    def __init__(self, Y, Q, kernel):
        self.Q = Q
        self.N, self.D = Y.shape
        self.kernel = kernel

        self.Ymean = np.mean(Y, axis=0)
        self.Ycentered = Y - self.Ymean
        self.YVar = MatrixVariable("Y", self.Ycentered)

        self.XVar = self.kernel.get_args()[0]
        self.XVar.val = dr.PCAreduction(self.Ycentered, Q)

        self.neglogliksym = GPLVMNegLogLik(kernel, self.YVar)
        self.functominimize = thx.FuncWithGrad(expr=self.neglogliksym.symbolic,
                                               args=symbols(self.neglogliksym.get_all_vars()),
                                               wrts=symbols(self.kernel.get_all_vars()))
        self.functominimize.set_args_values(values(self.neglogliksym.get_all_vars()))

        pfm.l_bfgs_b(self.functominimize, [self.XVar], maxiter=0)
        pfm.l_bfgs_b(self.functominimize, self.kernel.get_params(), maxiter=10)
        pfm.l_bfgs_b(self.functominimize, self.kernel.get_all_vars(), maxiter=100)

        self.K = self.kernel.function(self.XVar.val, self.XVar.val)
        self.Kinv = nt.NumpyLinalg.choleskyinv(self.K)

    def predict(self, xStar, fullCov=False):
        return eq.conditional_gaussian(self.XVar.val, self.Ymean, self.Ycentered, self.Kinv, self.kernel.function, xStar, fullCov=fullCov)



