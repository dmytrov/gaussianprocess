import numpy as np
import numerical.numpytheano as nt
import ml.gptheano.vecgpdm.equations as vceq
import matplotlibex.mlplot as plx
import matplotlib.pyplot as plt
from ml.gptheano.vecgpdm.enums import *


class Kernel(object):

    def __init__(self, diagonal_boost=False, ARD=False, 
                noscale=False, ndims=None, ns=nt.NumpyLinalg):
        super(Kernel, self).__init__()
        self.params = []
        self.ns = ns
        self.is_diagonal_boost = diagonal_boost
        self.is_ARD = ARD
        self.is_noscale = noscale
        self.ndims = ndims  # number of data points dimensions


    def gram_matrix(self, x1, x2, ns=nt.NumpyLinalg):
        raise NotImplementedError()


    def psi_stat_0(self, x_means, x_covars):
        raise NotImplementedError()


    def psi_stat_1(self, aug_z, x_means, x_covars):
        raise NotImplementedError()


    def psi_stat_2(self, aug_z, x_means, x_covars):
        raise NotImplementedError()


    def boost_gram_matrix(self, x1, x2, gm, ns):
        if x1 is x2 and self.is_diagonal_boost:
            print("Adding diagonal boost")
            return gm + 1.0e-3 * ns.identity(x1.shape[0])
        else:
            return gm


    def vec_if_not_ARD(self, v, ns=nt.NumpyLinalg):
        """Takes a scalar or a vector and returns a vector.
        """
        if self.is_ARD:
            return v
        else:
            return ns.zeros(self.ndims) + v

    
    def vec_if_ARD(self, v, ns=nt.NumpyLinalg):
        """Takes a scalar or a vector and returns a vector.
        """
        if not self.is_ARD:
            return v
        else:
            return ns.zeros(self.ndims) + v



class Linear_Kernel(Kernel):

    def __init__(self, ndims, kern_width, suffix, 
                ARD=False, noscale=False, ns=nt.NumpyLinalg):
        super(Linear_Kernel, self).__init__(ARD=ARD, noscale=noscale, 
                ndims=ndims, ns=ns)
        if self.is_ARD:
            var = ns.vector
        else:
            var = ns.scalar
        self.kernel_lambda_lin = var("kernel_lambda_lin"+suffix, 
                    self.vec_if_ARD(kern_width, ns=np), 
                    bounds=(1.0e-4, 1.0e4), 
                    tags=(VarTag.kernel_params))
        self.params.append(self.kernel_lambda_lin)
        self.kernel_lambda_lin_vec = self.vec_if_not_ARD(self.kernel_lambda_lin, ns=ns)

    
    def gram_matrix(self, x1, x2, ns=nt.NumpyLinalg):
        kernel_lambda_lin_vec = self.kernel_lambda_lin_vec
        if ns is nt.NumpyLinalg:
            kernel_lambda_lin_vec = self.ns.get_value(kernel_lambda_lin_vec)
        gm = vceq.Linear_ARD_Kern_diag_x_covar.gram_matrix(
                kernel_lambda_lin_vec, x1, x2, ns=ns)
        return self.boost_gram_matrix(x1, x2, gm, ns)

    
    def psi_stat_0(self, x_means, x_covars):
        return vceq.Linear_ARD_Kern_diag_x_covar.psi_stat_0(
                self.kernel_lambda_lin_vec,
                x_means, x_covars, ns=self.ns)

    
    def psi_stat_1(self, aug_z, x_means, x_covars):
        return vceq.Linear_ARD_Kern_diag_x_covar.psi_stat_1(
                self.kernel_lambda_lin_vec,
                aug_z, x_means, ns=self.ns)

    
    def psi_stat_2(self, aug_z, x_means, x_covars):
        return vceq.Linear_ARD_Kern_diag_x_covar.psi_stat_2(
                self.kernel_lambda_lin_vec,
                aug_z, x_means, x_covars, ns=self.ns)


class ARD_Linear_Kernel(Linear_Kernel):

    def __init__(self, ndims, kern_width, suffix, ns=nt.NumpyLinalg):
        super(ARD_Linear_Kernel, self).__init__(ARD=True, noscale=False,
                ndims=ndims, kern_width=kern_width, suffix=suffix, ns=ns)


class RBF_Kernel(Kernel):

    def __init__(self, ndims, kern_width, suffix,
                ARD=False, noscale=False, ns=nt.NumpyLinalg):
        super(RBF_Kernel, self).__init__(ARD=ARD, noscale=noscale, 
                ndims=ndims, ns=ns)
        if self.is_noscale:
            self.kernel_sigmasqrf = 1.0
        else:
            self.kernel_sigmasqrf = ns.scalar("kernel_sigmasqrf"+suffix, 1.0, 
                    bounds=(1.0e-3, 100.0), 
                    tags=(VarTag.kernel_params))  # scaling coefficient
            self.params.append(self.kernel_sigmasqrf)
        if self.is_ARD:
            var = ns.vector
        else:
            var = ns.scalar
        self.kernel_lambda_RBF = var("kernel_lambda_rbf"+suffix, 
                self.vec_if_ARD(kern_width, ns=np), 
                bounds=(1.0e-1, 100.0), 
                tags=(VarTag.kernel_params))  # width coefficient
        self.params.append(self.kernel_lambda_RBF)
        self.kernel_lambda_RBF_vec = self.vec_if_not_ARD(self.kernel_lambda_RBF, ns=ns)


    def gram_matrix(self, x1, x2, ns=nt.NumpyLinalg):
        kernel_sigmasqrf = self.kernel_sigmasqrf
        kernel_lambda = self.kernel_lambda_RBF_vec
        if ns is nt.NumpyLinalg:
            if not self.is_noscale:
                kernel_sigmasqrf = self.ns.evaluate(kernel_sigmasqrf)
            kernel_lambda = self.ns.evaluate(kernel_lambda)
        gm = vceq.RBF_ARD_Kern_diag_x_covar.gram_matrix(kernel_sigmasqrf, 
                kernel_lambda,
                x1, x2, ns=ns)
        return self.boost_gram_matrix(x1, x2, gm, ns)


    def psi_stat_0(self, x_means, x_covars):
        return vceq.RBF_ARD_Kern_diag_x_covar.psi_stat_0(self.kernel_sigmasqrf, 
                x_means, ns=self.ns)

    def psi_stat_1(self, aug_z, x_means, x_covars):
        return vceq.RBF_ARD_Kern_diag_x_covar.psi_stat_1(self.kernel_sigmasqrf,
                self.kernel_lambda_RBF_vec,
                aug_z, x_means, x_covars, ns=self.ns)


    def psi_stat_2(self, aug_z, x_means, x_covars):
        return vceq.RBF_ARD_Kern_diag_x_covar.psi_stat_2(self.kernel_sigmasqrf,
                self.kernel_lambda_RBF_vec,
                aug_z, x_means, x_covars, ns=self.ns)



class RBF_Kernel_noscale(RBF_Kernel):

    def __init__(self, ndims, kern_width, suffix, ns=nt.NumpyLinalg):
        super(RBF_Kernel_noscale, self).__init__(ARD=False, noscale=True,
                ndims=ndims, kern_width=kern_width, suffix=suffix, ns=ns)



class ARD_RBF_Kernel(RBF_Kernel):

    def __init__(self, ndims, kern_width, suffix, ns=nt.NumpyLinalg):
        super(ARD_RBF_Kernel, self).__init__(ARD=True, noscale=False,
                ndims=ndims, kern_width=kern_width, suffix=suffix, ns=ns)



class ARD_RBF_Kernel_noscale(RBF_Kernel):

    def __init__(self, ndims, kern_width, suffix, ns=nt.NumpyLinalg):
        super(ARD_RBF_Kernel_noscale, self).__init__(ARD=True, noscale=True,
                ndims=ndims, kern_width=kern_width, suffix=suffix, ns=ns)



class RBF_plus_Linear_Kernel(Kernel):

    def __init__(self, ndims, kern_width, suffix,
                ARD=False, noscale=False, ns=nt.NumpyLinalg):
        super(RBF_plus_Linear_Kernel, self).__init__(ARD=ARD, noscale=noscale, 
                ndims=ndims, ns=ns)
        if self.is_noscale:
            self.kernel_sigmasqrf = 1.0
        else:
            self.kernel_sigmasqrf = ns.scalar("kernel_sigmasqrf"+suffix, 1.0, 
                    bounds=(1.0e-3, 100.0), 
                    tags=(VarTag.kernel_params))  # scaling coefficient
            self.params.append(self.kernel_sigmasqrf)
        if self.is_ARD:
            var = ns.vector
        else:
            var = ns.scalar
        self.kernel_lambda_rbf = var("kernel_lambda_rbf"+suffix, 
                self.vec_if_ARD(kern_width, ns=np), 
                bounds=(1.0e-1, 100.0), 
                tags=(VarTag.kernel_params))  # width coefficient
        self.kernel_lambda_lin = var("kernel_lambda_lin"+suffix, 
                self.vec_if_ARD(100.0 * kern_width), 
                bounds=(1.0e-4, 1.0e4), 
                tags=(VarTag.kernel_params))
        self.params.append(self.kernel_lambda_rbf)
        self.params.append(self.kernel_lambda_lin)
        self.kernel_lambda_rbf_vec = self.vec_if_not_ARD(self.kernel_lambda_rbf, ns=ns)
        self.kernel_lambda_lin_vec = self.vec_if_not_ARD(self.kernel_lambda_lin, ns=ns)


    def gram_matrix(self, x1, x2, ns=nt.NumpyLinalg):
        kernel_sigmasqrf = self.kernel_sigmasqrf
        kernel_lambda_rbf = self.kernel_lambda_rbf_vec
        kernel_lambda_lin = self.kernel_lambda_lin_vec
        if ns is nt.NumpyLinalg:
            if not self.is_noscale:
                kernel_sigmasqrf = self.ns.evaluate(kernel_sigmasqrf)
            kernel_lambda_rbf = self.ns.get_value(kernel_lambda_rbf)
            kernel_lambda_lin = self.ns.get_value(kernel_lambda_lin)
        gm = vceq.RBF_plus_Linear_ARD_Kern_diag_x_covar.gram_matrix(
                sigmasqrf=kernel_sigmasqrf, 
                lambdas_rbf=kernel_lambda_rbf,
                lambdas_linear=kernel_lambda_lin,
                x1=x1, x2=x2, ns=ns)
        return self.boost_gram_matrix(x1, x2, gm, ns)


    def psi_stat_0(self, x_means, x_covars):
        return vceq.RBF_plus_Linear_ARD_Kern_diag_x_covar.psi_stat_0(
                sigmasqrf=self.kernel_sigmasqrf, 
                lambdas_linear=self.kernel_lambda_lin,
                x_means=x_means, x_covars=x_covars, ns=self.ns)


    def psi_stat_1(self, aug_z, x_means, x_covars):
        return vceq.RBF_plus_Linear_ARD_Kern_diag_x_covar.psi_stat_1(
                sigmasqrf=self.kernel_sigmasqrf,
                lambdas_rbf=self.kernel_lambda_rbf,
                lambdas_linear=self.kernel_lambda_lin,
                aug_z=aug_z, x_means=x_means, x_covars=x_covars, ns=self.ns)


    def psi_stat_2(self, aug_z, x_means, x_covars):
        return vceq.RBF_plus_Linear_ARD_Kern_diag_x_covar.psi_stat_2(
                sigmasqrf=self.kernel_sigmasqrf,
                lambdas_rbf=self.kernel_lambda_rbf,
                lambdas_linear=self.kernel_lambda_lin,
                aug_z=aug_z, x_means=x_means, x_covars=x_covars, ns=self.ns)


class ARD_RBF_plus_linear_Kernel(RBF_plus_Linear_Kernel):

    def __init__(self, ndims, kern_width, suffix, ns=nt.NumpyLinalg):
        super(ARD_RBF_plus_linear_Kernel, self).__init__(ARD=True, noscale=False,
                ndims=ndims, kern_width=kern_width, suffix=suffix, ns=ns)


class ARD_RBF_plus_linear_Kernel_noscale(RBF_plus_Linear_Kernel):

    def __init__(self, ndims, kern_width, suffix, ns=nt.NumpyLinalg):
        super(ARD_RBF_plus_linear_Kernel_noscale, self).__init__(ARD=True, noscale=True,
                ndims=ndims, kern_width=kern_width, suffix=suffix, ns=ns)


class PoE_Kernel(Kernel):

    def __init__(self, Kobjs, alphas, ns=nt.NumpyLinalg):
        super(PoE_Kernel, self).__init__()
        self.params = []
        self.ns = ns
        self.Kobjs = Kobjs
        self.alphas = alphas
        

    def gram_matrix(self, x1s, x2s, ns=nt.NumpyLinalg):
        M = len(self.Kobjs)
        alphas = self.alphas
        if ns is nt.NumpyLinalg:
            alphas = self.ns.evaluate(self.alphas)
        
        alphas_inv = 1.0 / alphas
        alpha_joint_inv = ns.sum(alphas_inv)  
        alpha_joint = 1.0 / alpha_joint_inv 
        
        # Partial kernels
        Kxxs = [self.Kobjs[i].gram_matrix(x1s[i], x2s[i], ns=ns) for i in range(M)]

        # Full PoE kernel
        res = 0
        for j in range(M):
            res += (alpha_joint / alphas[j])**2 * Kxxs[j]
        
        # Add diagonal noise if needed
        if x1s is x2s:
            res += ns.identity(res.shape[0]) * alpha_joint

        return res


class Noise_Kernel(Kernel):

    def __init__(self, sigmasqr, ns=nt.NumpyLinalg):
        super(Noise_Kernel, self).__init__()
        self.params = []
        self.ns = ns
        self.sigmasqr = sigmasqr
        

    def gram_matrix(self, x1, x2, ns=nt.NumpyLinalg):
        sigmasqr = self.sigmasqr
        if ns is nt.NumpyLinalg:
            sigmasqr = self.ns.evaluate(self.sigmasqr)
        
        if x1 is x2:
            res = ns.identity(x1.shape[0]) * sigmasqr
        else:
            res = ns.zeros([x1.shape[0], x2.shape[0]])

        return res


class Sum_Kernel(Kernel):

    def __init__(self, Kobjs, ns=nt.NumpyLinalg):
        super(Sum_Kernel, self).__init__()
        self.params = []
        self.ns = ns
        self.Kobjs = Kobjs
        

    def gram_matrix(self, x1, x2, ns=nt.NumpyLinalg):
        M = len(self.Kobjs)
        
        # Partial kernels
        res = 0
        for i in range(M):
            res += self.Kobjs[i].gram_matrix(x1, x2, ns=ns)
        
        return res