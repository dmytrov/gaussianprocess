import numerical.theanoext as thx
import numerical.numpyext.dimreduction as dr
from ml.gptheano.kernels import *
import numerical.theanoext.parametricfunctionminimization as pfm


class GPDMNegLogLik(ParametricFunction):
    def __init__(self, dynamicsOrder, kernelLVM, kernelDM, XVar, YVar):
        super(GPDMNegLogLik, self).__init__()
        if XVar is None:
            XVar = kernelLVM.get_args()[0]
        self.YVar = YVar
        self.XVar = XVar
        self.N, self.Q = self.XVar.val.shape
        self._params = [self.YVar]
        self.kernelLVM = kernelLVM
        self.kernelDM = kernelDM
        self._children = [kernelLVM, kernelDM]
        self.dynamicsOrder = dynamicsOrder
        self.symbolic = -eq.gplvm_gpdm_loglik(self.Q, self.N, dynamicsOrder,
                                                self.kernelLVM.symbolic,
                                                self.kernelDM.symbolic,
                                                self.XVar.symbolic,
                                                self.YVar.symbolic,
                                                ns=nt.TheanoLinalg)
        self.function = lambda X: -eq.gplvm_gpdm_loglik(self.Q, self.N, self.dynamicsOrder,
                                     self.kernelLVM.function(X, X),
                                     self.kernelDM.function(X, X),
                                     X,
                                     self.YVar.val,
                                     ns=nt.NumpyLinalg)

class GPDM(object):
    def __init__(self, dynamicsOrder, Y, Q, kernelLVM, kernelDM):
        self.dynamicsOrder = dynamicsOrder
        self.Q = Q
        self.N, self.D = Y.shape
        self.kernelLVM = kernelLVM
        self.kernelDM = kernelDM

        self.Ymean = np.mean(Y, axis=0)
        self.Ycentered = Y - self.Ymean
        self.YVar = MatrixVariable("Y", self.Ycentered)

        self.XVar = self.kernelLVM.get_args()[0]
        self.XVar.val = dr.PCAreduction(self.Ycentered, Q)

        self.neglogliksym = GPDMNegLogLik(self.dynamicsOrder, self.kernelLVM, self.kernelDM, self.XVar, self.YVar)
        self.functominimize = thx.FuncWithGrad(expr=self.neglogliksym.symbolic,
                                               args=symbols(self.neglogliksym.get_all_vars()))
        self.functominimize.set_args_values(values(self.neglogliksym.get_all_vars()))

        print("Staring negative loglikelohood: ", self.neglogliksym.function(self.XVar.val))
        pfm.l_bfgs_b(self.functominimize, [self.XVar], maxiter=0)
        pfm.l_bfgs_b(self.functominimize, self.kernelLVM.get_params() + self.kernelDM.get_params(), maxiter=10)
        pfm.l_bfgs_b(self.functominimize, self.kernelLVM.get_all_vars() + self.kernelDM.get_all_vars(), maxiter=100)

        self.KLVM = self.kernelLVM.function(self.XVar.val, self.XVar.val)
        self.KLVMinv = nt.NumpyLinalg.choleskyinv(self.KLVM)
        self.KDM = self.kernelDM.function(self.XVar.val, self.XVar.val)
        self.KDMinv = nt.NumpyLinalg.choleskyinv(self.KDM)

    def X_to_Y(self, xStar, fullCov=False):
        return eq.conditional_gaussian(self.XVar.val,
                                       self.Ymean, self.Ycentered,
                                       self.KLVMinv, self.kernelLVM.function,
                                       xStar, fullCov=fullCov)

    def generate_dynamics(self, count=10, x0=None):
        if x0 is None:
            x0 = self.XVar.val[0:self.dynamicsOrder, :]
        assert x0.shape[0] == self.dynamicsOrder
        Xpath = x0
        Xtplus = eq.Xtplus(self.XVar.val, self.dynamicsOrder)
        for i in range(count):
            Xt, Vart = eq.conditional_gaussian(self.XVar.val, 0, Xtplus,
                                         self.KDMinv, self.kernelDM.function,
                                         np.vstack((Xpath[-self.dynamicsOrder:, :], Xpath[-1, :])),
                                         fullCov=False, order=self.dynamicsOrder)
            Xpath = np.vstack((Xpath, Xt))
        YPath = self.X_to_Y(Xpath)
        return Xpath, YPath

    def predict(self, xStar, fullCov=False):
        return eq.conditional_gaussian(self.XVar.val, self.Ymean, self.Ycentered, self.KLVMinv, self.kernelLVM.function, xStar, fullCov=fullCov)



