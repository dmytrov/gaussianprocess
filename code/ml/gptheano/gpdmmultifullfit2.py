import numpy as np
import numerical.theanoext as thx
import numerical.numpyext.dimreduction as dr
from ml.gptheano.kernels import *
import numerical.theanoext.parametricfunctionminimization as pfm


class GPDMMultiNegLogLik2(ParametricFunction):
    def __init__(self, dynamicsOrder, kernelLVM, kernelDM, XVar, YVar, sequencesindexes):
        super(GPDMMultiNegLogLik2, self).__init__()
        self.dynamicsOrder = dynamicsOrder
        self.kernelLVM = kernelLVM
        self.kernelDM = kernelDM
        self.YVar = YVar
        self.XVar = XVar
        self.N, self.Q = self.XVar.val.shape
        self.sequencesindexes = sequencesindexes

        self.xt0indexes, self.xtminusindexes, self.xtplusindexes = eq.xt0_xtminus_xtplus_indexes(self.sequencesindexes, self.dynamicsOrder)
        self.xt0indexesVar = IntVectorVariable("Xt0Indexes", self.xt0indexes)
        self.xtplusindexesVar = IntVectorVariable("XtplusIndexes", self.xtplusindexes)
        self.xtoxtminusmap = XtoXtminusMap(XVar, self.xtminusindexes)
        self.kernelDMobj = kernelDM(self.xtoxtminusmap, None)
        self.kernelLVMobj = kernelLVM([self.XVar], None)
        self._children = [self.kernelLVMobj, self.kernelDMobj] + self.xtoxtminusmap
        self._params = [self.XVar, self.YVar]
        self._consts = [self.xt0indexesVar] + [self.xtplusindexesVar]
        self.symbolic = -eq.gplvm_gpdm_loglik_2(self.Q, dynamicsOrder,
                                                self.kernelLVMobj.symbolic,
                                                self.kernelDMobj.symbolic,
                                                self.XVar.symbolic,
                                                self.YVar.symbolic,
                                                self.xt0indexesVar.symbolic,
                                                self.xtplusindexesVar.symbolic,
                                                ns=nt.TheanoLinalg)
        self.function = lambda X: -eq.gplvm_gpdm_loglik_2(self.Q, self.dynamicsOrder,
                                     self.kernelLVMobj.function([X], None),
                                     self.kernelDMobj.function([xmap.function(X) for xmap in self.xtoxtminusmap], None),
                                     X,
                                     self.YVar.val,
                                     self.xt0indexesVar.val,
                                     self.xtplusindexesVar.val,
                                     ns=nt.NumpyLinalg)

class GPDMMulti2(object):
    """
    Multi-trial GPDM
    """
    def __init__(self, kernelLVM, kernelDM, Ysequences=[np.ones([5, 3]), np.ones([6, 3])], Q=2, dynamicsOrder=2):
        """
        Constructor
        :param kernelLVM: X->Y GP mapping kernel object factory
        :param kernelDM: X_tminus->X_tplus mapping kernel object factory
        :param Ysequences: NxD matrix of observation points or list of N_ixD observation points matrices
        :param Q: latent dimensinoality
        :param dynamicsOrder: dynamics order (1 or 2)
        :return:
        """
        if not isinstance(Ysequences, list):
            Ysequences = [Ysequences]

        sequencesindexes = eq.sequences_indexes(Ysequences)
        self.Y = np.vstack(Ysequences)
        self.dynamicsOrder = dynamicsOrder
        self.Q = Q
        self.N, self.D = self.Y.shape
        self.kernelLVM = kernelLVM
        self.kernelDM = kernelDM

        self.Ymean = np.mean(self.Y, axis=0)
        self.Ycentered = self.Y - self.Ymean
        self.YVar = MatrixVariable("Y", self.Ycentered)

        self.XVar =  MatrixVariable("X", self.Ycentered)
        self.XVar.val = dr.PCAreduction(self.Ycentered, Q)


        self.neglogliksym = GPDMMultiNegLogLik2(self.dynamicsOrder, self.kernelLVM, self.kernelDM, self.XVar, self.YVar, sequencesindexes)
        self.functominimize = thx.FuncWithGrad(expr=self.neglogliksym.symbolic,
                                               args=symbols(self.neglogliksym.get_all_vars()))
        self.functominimize.set_args_values(values(self.neglogliksym.get_all_vars()))

        print("Staring negative loglikelohood, NumPy: ", self.neglogliksym.function(self.XVar.val))
        print("Staring negative loglikelohood, Theano: ", self.functominimize.get_func_value())
        pfm.l_bfgs_b(self.functominimize, [self.XVar], maxiter=1)
        pfm.l_bfgs_b(self.functominimize, self.neglogliksym.kernelLVMobj.get_params() + self.neglogliksym.kernelDMobj.get_params(), maxiter=10)
        pfm.l_bfgs_b(self.functominimize, self.neglogliksym.kernelLVMobj.get_params() + self.neglogliksym.kernelDMobj.get_params() + [self.XVar], maxiter=100)
        print("Final negative loglikelohood, NumPy: ", self.neglogliksym.function(self.XVar.val))
        print("Final negative loglikelohood, Theano: ", self.functominimize.get_func_value())

        self.KLVM = self.neglogliksym.kernelLVMobj.function([self.XVar.val], None)
        self.KLVMinv = nt.NumpyLinalg.choleskyinv(self.KLVM)
        self.KDM = self.neglogliksym.kernelDMobj.function([self.XVar.val[i] for i in self.neglogliksym.xtminusindexes], None)
        self.KDMinv = nt.NumpyLinalg.choleskyinv(self.KDM)

    def X_to_Y(self, xStar, fullCov=False):
        return eq.conditional_gaussian(self.XVar.val,
                                       self.Ymean, self.Ycentered,
                                       self.KLVMinv, self.neglogliksym.kernelLVMobj.function,
                                       xStar, fullCov=fullCov)

    def generate_dynamics(self, count=10, x0=None):
        if x0 is None:
            x0 = self.XVar.val[0:self.dynamicsOrder, :]
        assert x0.shape[0] == self.dynamicsOrder
        Xpath = x0
        Xtminus = [xmap.function(self.XVar.val) for xmap in self.neglogliksym.xtoxtminusmap]
        Xtplus = self.XVar.val[self.neglogliksym.xtplusindexes]
        _, xtminusindexes, _ = eq.xt0_xtminus_xtplus_indexes([[-3, -2, -1]], self.dynamicsOrder)
        for i in range(count):
            Xt, Vart = eq.conditional_gaussian(Xtminus, 0, Xtplus,
                                         self.KDMinv,
                                         lambda X1, X2: self.neglogliksym.kernelDMobj.function(X1, X2),
                                         [np.vstack([Xpath[-self.dynamicsOrder:, :], Xpath[-1, :]])[ind] for ind in xtminusindexes],
                                         fullCov=False, order=self.dynamicsOrder)
            Xpath = np.vstack((Xpath, Xt))
        YPath = self.X_to_Y(Xpath)
        return Xpath, YPath

    def predict(self, xStar, fullCov=False):
        if not isinstance(xStar, np.ndarray):
            xStar = np.array(xStar)
        return eq.conditional_gaussian(self.XVar.val, self.Ymean, self.Ycentered, self.KLVMinv, self.neglogliksym.kernelLVMobj.function, xStar, fullCov=fullCov)





