import numpy as np
import theano
import numerical.theanoext as thx
import numerical.numpyext.dimreduction as dr
from ml.gptheano.kernels import *
import numerical.theanoext.parametricfunctionminimization as pfm


class Part(object):
    pass
class ModelParams(object):
    pass

class CGPDMLogLik(ParametricFunction):
    def __init__(self, parts=[Part(), Part()], model_prams=ModelParams()):
        super(CGPDMLogLik, self).__init__()
        self.parts = parts
        self.model_prams = model_prams
        self.coupled_kerns = CouplingKern([part.DM_kernel for part in self.parts])

        self._consts = [model_prams.xtplusindexes_var, model_prams.xt0indexes_var]
        self._args = [part.X_var for part in parts] + [part.Y_var for part in parts]
        self._children = [self.coupled_kerns] + [part.LVM_kernel for part in parts]
        for part in parts:
            self._children.extend(part.x_to_xminus_map)
        self.symbolic = 0
        for part, coupled_kern_sym in zip(parts, self.coupled_kerns.symbolic):
            self.symbolic -= eq.gplvm_gpdm_loglik_2(part.Q,
                                                    model_prams.dynamics_order,
                                                    part.LVM_kernel.symbolic,
                                                    coupled_kern_sym,
                                                    part.X_var.symbolic,
                                                    part.Y_var.symbolic,
                                                    model_prams.xt0indexes_var.symbolic,
                                                    model_prams.xtplusindexes_var.symbolic,
                                                    ns=nt.TheanoLinalg)
        self.function = self._function

    def _function(self):
        DM_partials = [part.DM_kernel.function([x_to_xminus.function(part.X_var.val) for x_to_xminus in part.x_to_xminus_map], None) for part in self.parts]
        coupled_kerns = self.coupled_kerns.function(DM_partials, X1_is_X2=True)
        res = 0
        for part, coupled_kern in zip(self.parts, coupled_kerns):
            res -= eq.gplvm_gpdm_loglik_2(part.Q,
                                          self.model_prams.dynamics_order,
                                          part.LVM_kernel.function(part.X_var.val, None),
                                          coupled_kern,
                                          part.X_var.val,
                                          part.Y_var.val,
                                          self.model_prams.xt0indexes_var.val,
                                          self.model_prams.xtplusindexes_var.val,
                                          ns=nt.NumpyLinalg)
        return res

class Data(object):
    def __init__(self,
                 Y_sequences):  # =[np.ones([5, 3]), np.ones([6, 3])]):
        if not isinstance(Y_sequences, list):
            Y_sequences = [Y_sequences]
        self.Y_sequences = Y_sequences
        self.Y = np.vstack(Y_sequences)
        self.sequences_indexes = eq.sequences_indexes(Y_sequences)
        self.N, self.D = self.Y.shape


class ModelParams(object):
    def __init__(self,
                 data,  # =Data(),
                 Qs,  # =[2, 2, 2],
                 dynamics_order,  # =2,
                 parts_IDs,  # =[0, 0, 1, 1, 2, 2],
                 LVM_kernel_factory,  # =None,
                 DM_kernel_factory):  # =None):
        self.Qs, self.dynamics_order, self.parts_IDs, self.LVM_kernel_factory, self.DM_kernel_factory = \
            Qs, dynamics_order, parts_IDs, LVM_kernel_factory, DM_kernel_factory

        assert len(parts_IDs) == data.D
        assert min(parts_IDs) == 0
        self.nparts = max(parts_IDs) + 1
        self.parts_indexes = [[j for j, index in enumerate(parts_IDs) if i == index] for i in range(self.nparts)]
        self.xt0indexes, self.xtminusindexes, self.xtplusindexes = eq.xt0_xtminus_xtplus_indexes(data.sequences_indexes, self.dynamics_order)
        self.xt0indexes_var = IntVectorVariable("Xt0Indexes", self.xt0indexes)
        self.xtplusindexes_var = IntVectorVariable("XtplusIndexes", self.xtplusindexes)


class Part(object):
    def __init__(self,
                 ID,  # ="0",
                 Y,  # =np.ones([9, 3]),
                 Q,  # =2,
                 model_prams):  # =ModelParams()):
        self.ID = ID
        self.Y = Y
        self.Q = Q
        self.model_prams = model_prams

        self.Y_mean = np.mean(self.Y, axis=0)
        self.Y_centered = self.Y - self.Y_mean
        self.Y_var = MatrixVariable("Y"+ID, self.Y_centered)
        self.X_var = MatrixVariable("X"+ID, dr.PCAreduction(self.Y_centered, Q))
        self.LVM_kernel = model_prams.LVM_kernel_factory([self.X_var], None)
        self.x_to_xminus_map = XtoXtminusMap(self.X_var, model_prams.xtminusindexes)
        self.DM_kernel = model_prams.DM_kernel_factory(self.x_to_xminus_map, None)
        #self.coupled_kernel = None


class CGPDMMulti(object):
    """
    Multi-trial CGPDM
    """
    def __init__(self,
                 data,  # =Data(),
                 model_params):  # =ModelParams()):
        self.data = data
        self.model_params = model_params

        self.parts = []
        for i in range(model_params.nparts):
            self.parts.append(Part(str(i), data.Y[:, model_params.parts_indexes[i]], model_params.Qs[i], model_params))

        self.neglogliksym = CGPDMLogLik(self.parts, model_params)

        self.functominimize = thx.FuncWithGrad(expr=self.neglogliksym.symbolic,
                                               args=symbols(self.neglogliksym.get_all_vars()))
        self.functominimize.set_args_values(values(self.neglogliksym.get_all_vars()))

        print("Staring negative loglikelohood, NumPy: ", self.neglogliksym.function())
        print("Staring negative loglikelohood, Theano: ", self.functominimize.get_func_value())
        x_vars = [part.X_var for part in self.parts]
        kernels_params = self.neglogliksym.coupled_kerns.get_params()
        for part in self.parts:
            kernels_params.extend(part.LVM_kernel.get_params())
        pfm.l_bfgs_b(self.functominimize, x_vars, maxiter=1)
        pfm.l_bfgs_b(self.functominimize, kernels_params, maxiter=10)
        print("Coupling matrix: ", self.neglogliksym.coupled_kerns.log_sigma_sqrs_var.val)
        pfm.l_bfgs_b(self.functominimize, kernels_params + x_vars, maxiter=10)
        print("Final negative loglikelohood, NumPy: ", self.neglogliksym.function())
        print("Final negative loglikelohood, Theano: ", self.functominimize.get_func_value())
        print("Coupling matrix: ", self.neglogliksym.coupled_kerns.log_sigma_sqrs_var.val)


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



