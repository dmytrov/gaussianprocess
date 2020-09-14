import os
import time
import numpy as np
import numerical.numpytheano as nt
import numerical.numpyext.linalg as ntla
import numerical.numpyext as npx
import numerical.numpyext.dimreduction as dr
import theano
import ml.gp.vcgpdm.equations as vceq
import ml.gp.equations as gpeq
import numerical.theanoext as thx
import numerical.theanoext.parametricfunctionminimization as pfm
import numerical.theanoext.parametricfunction as pf
import matplotlibex.mlplot as plx
import matplotlib.pyplot as plt


def match(var, ns=nt.NumpyLinalg):
    if ns==nt.NumpyLinalg:
        return var.val 
    elif ns==nt.TheanoLinalg:
        return var.symbolic
    else:
        raise NotImplementedError()


class PartData(object):
    def __init__(self,
                 Y_value,
                 Y, 
                 ns=nt.NumpyLinalg): 
        self.Y_value = Y_value
        self.N, self.D = self.Y_value.shape
        self.Y_mean_value = np.mean(self.Y_value, axis=0)
        self.Y_centered_value = self.Y_value - self.Y_mean_value

        self.Y = Y
        self.Y_mean = ns.mean(self.Y, axis=0)
        self.Y_centered = self.Y - self.Y_mean

class PartParams(object):
    def __init__(self,
                 data,  # =PartData(),
                 Q,  # =2, latent dimensionality
                 xt0indexes_var,
                 xtminusindexes_vars,
                 xtplusindexes_var,
                 xt0indexes,
                 xtminusindexes,
                 xtplusindexes,
                 M,  # =10, number of inducing mappings
                 ID,
                 suffix,  # string description
                 init_func
                 ):
        self.data = data
        self.Q = Q
        self.dynamics_order = 2
        self.xt0indexes_var, self.xtminusindexes_vars, self.xtplusindexes_var = xt0indexes_var, xtminusindexes_vars, xtplusindexes_var
        self.xt0indexes, self.xtminusindexes, self.xtplusindexes = xt0indexes, xtminusindexes, xtplusindexes
        if isinstance(M, (list, tuple)):
            assert len(M) == 2
            self.dyn_M = M[0]
            self.lvm_M = M[1]
        else:
            self.dyn_M = M
            self.lvm_M = M
        self.ID = ID
        self.suffix = suffix 
        self.init_func = init_func


class ModelData(object):
    def __init__(self,
                 Y_sequences,
                 ns=nt.NumpyLinalg):  # =[np.ones([5, 3]), np.ones([6, 3])]):
        if not isinstance(Y_sequences, list):
            Y_sequences = [Y_sequences]
        self.Y_sequences = Y_sequences
        self.Y_var = pf.MatrixVariable("Y", np.vstack(Y_sequences))
        self.Y = match(self.Y_var, ns)
        self.sequences_indexes = gpeq.sequences_indexes(Y_sequences)
        self.N, self.D = self.Y_var.val.shape


class ModelParams(object):
    def __init__(self,
                 data,  # =ModelData(),
                 Qs,  # =[2, 2, 2], latent dimensions
                 parts_IDs,  # =[0, 0, 1, 1, 2, 2]
                 M,  # =10, number of inducing mappings
                 init_func=None,
                 ns=nt.NumpyLinalg):
        self.data = data
        self.Qs = Qs
        self.parts_IDs = parts_IDs
        self.dynamics_order = 2
        self.M = M
        self.init_func = init_func

        assert len(parts_IDs) == data.D
        assert min(parts_IDs) == 0
        self.nparts = max(parts_IDs) + 1
        self.parts_indexes_var = [pf.IntVectorVariable("parts_indexes", np.array([j for j, index in enumerate(parts_IDs) if i == index])) for i in range(self.nparts)]
        self.parts_indexes = [match(piv, ns) for piv in self.parts_indexes_var]

        xt0indexes, xtminusindexes, xtplusindexes = gpeq.xt0_xtminus_xtplus_indexes(data.sequences_indexes, self.dynamics_order)
        self.xt0indexes_var = pf.IntVectorVariable("xt0indexes", xt0indexes)
        self.xt0indexes = match(self.xt0indexes_var, ns)
        self.xtminusindexes_vars = [pf.IntVectorVariable("xtminusindexes", i) for i in xtminusindexes]
        self.xtminusindexes = [match(i, ns) for i in self.xtminusindexes_vars]
        self.xtplusindexes_var = pf.IntVectorVariable("xtplusindexes", xtplusindexes)
        self.xtplusindexes = match(self.xtplusindexes_var, ns)

    def make_part_params(self, part_ID, ns=nt.NumpyLinalg):
        res = PartParams(PartData(self.data.Y_var.val[:, self.parts_indexes_var[part_ID].val], 
                                  self.data.Y[:, self.parts_indexes[part_ID]], 
                                  ns),
                         self.Qs[part_ID],
                         self.xt0indexes_var,
                         self.xtminusindexes_vars,
                         self.xtplusindexes_var,
                         self.xt0indexes,
                         self.xtminusindexes,
                         self.xtplusindexes,
                         self.M,
                         part_ID,
                         "_" + str(part_ID),
                         self.init_func)
        return res

    def make_parts_params(self):
        return [self.make_part_params(part_ID) for part_ID in range(self.nparts)]


class VCGPDMPart(pf.ParametricFunction):
    def __init__(self, 
                 partparams,  # =PartParams()
                 ns=nt.NumpyLinalg):
        super(VCGPDMPart, self).__init__()
        self.ns = ns
        self.params = partparams
        Q = self.params.Q
        N = self.params.data.N
        suffix = self.params.suffix
        self.kern_type = vceq.RBF_ARD_kern_diag_x_covar

        # Mean step length in the latent space
        
        if self.params.init_func is None:
            init_x = np.sqrt(self.params.data.Y_centered_value.shape[0]) * dr.PCAreduction(self.params.data.Y_centered_value, Q)
        else:
            init_x = self.params.init_func(self.params.data.Y_centered_value, Q, self.params.ID)
        self.x_means_var = pf.MatrixVariable("x_means"+suffix, init_x, bounds=(-5.0, 5.0))
        step = self.x_means_var.val[self.params.xtminusindexes_vars[1].val, :] - self.x_means_var.val[self.params.xtminusindexes_vars[0].val, :]
        step = np.mean(np.sqrt(step * step))
        kern_width = 3.5 * step
        self.log_x_covars_diags_var = pf.MatrixVariable("log_x_covars_diags"+suffix, np.log(0.1 * step + np.zeros([N, Q])), bounds=(-6.0, 2.0))
        
        # Dynamics
        self.dyn_kernparam_alpha = None  # set by the coupling 
        self.dyn_log_kernparam_sigmasqrf_var = pf.ScalarVariable("dyn_log_kernparam_sigmasqrf"+suffix, np.log(1.0), bounds=(-2.0, 2.0))  # scaling coefficient
        self.dyn_log_kernparam_lambda1_var = pf.ScalarVariable("dyn_log_kernparam_lambda1"+suffix, np.log(kern_width), bounds=(-2.0, 2.0))  # for x_t-1
        self.dyn_log_kernparam_lambda2_var = pf.ScalarVariable("dyn_log_kernparam_lambda2"+suffix, np.log(kern_width), bounds=(-2.0, 2.0))  # for x_t-2
        
        nmappings = self.params.xtplusindexes_var.val.size
        aug_indexes = np.arange(0, nmappings, int(nmappings/self.params.dyn_M))[:self.params.dyn_M]
        dyn_aug_in = np.hstack([self.x_means_var.val[self.params.xtminusindexes_vars[0].val[aug_indexes], :], 
                                self.x_means_var.val[self.params.xtminusindexes_vars[1].val[aug_indexes], :]])
        #dyn_aug_in += 0.1 * np.reshape(np.random.normal(size=dyn_aug_in.size), dyn_aug_in.shape)
        self.dyn_aug_in_var = pf.MatrixVariable("dyn_aug_in"+suffix, dyn_aug_in, bounds=(-10.0, 10.0))
        self.dyn_psi1 = None  
        self.dyn_psi2 = None  
        self.dyn_Kaug = None  
        self.dyn_Kauginv = None  

        # (GP) Latent Variable Model
        nmappings = self.params.xtplusindexes_var.val.size
        aug_indexes = np.arange(0, nmappings, int(nmappings/self.params.lvm_M))[:self.params.lvm_M]
        lvm_aug_in = self.x_means_var.val[self.params.xtminusindexes_vars[0].val[aug_indexes], :]
        #lvm_aug_in += 0.1 * np.random.normal(size=lvm_aug_in.shape[0])[:, np.newaxis]
        self.lvm_aug_in_var = pf.MatrixVariable("lvm_aug_in"+suffix, lvm_aug_in, bounds=(-10.0, 10.0))
        self.lvm_log_kernparam_beta_var = pf.ScalarVariable("lvm_log_kernparam_beta"+suffix, np.log(2.0), bounds=(-2.0, 2.0))  # !!! INVERSE !!! of the x->y mapping noise
        self.lvm_log_kernparam_sigmasqrf_var = pf.ScalarVariable("lvm_log_kernparam_sigmasqrf"+suffix, np.log(1.0), bounds=(-2.0, 2.0))  # scaling coefficient
        self.lvm_log_kernparam_lambda_var = pf.ScalarVariable("lvm_log_kernparam_lambda"+suffix, np.log(kern_width), bounds=(-2.0, 2.0))
        
        [self.x_means,
         self.x_covars_diags,
         self.dyn_xt_means,
         self.dyn_xt_covars,
         self.dyn_Kaug_partial, 
         self.dyn_psi0,
         self.dyn_psi1_partial,
         self.dyn_psi2_partial,
         self.lvm_kernparam_beta,
         self.lvm_Kaug,
         self.lvm_psi0,
         self.lvm_psi1,
         self.lvm_psi2] = self.compute_dyn_lvm_vars(ns=ns)

        self._params = [self.x_means_var,
                        self.log_x_covars_diags_var,
                        self.dyn_log_kernparam_sigmasqrf_var,
                        self.dyn_log_kernparam_lambda1_var,
                        self.dyn_log_kernparam_lambda2_var,
                        self.dyn_aug_in_var,
                        self.lvm_log_kernparam_beta_var,
                        self.lvm_log_kernparam_sigmasqrf_var,
                        self.lvm_log_kernparam_lambda_var,
                        self.lvm_aug_in_var
                        ]
        
        # pp_ - for the posterior-predictive 
        self.pp_x_means = None
        self.pp_dyn_xt_means = None
        self.pp_dyn_xt_covars = None
        self.pp_dyn_Kaug_partial = None
        self.pp_dyn_psi1_partial = None
        self.pp_dyn_psi2_partial = None
        self.pp_dyn_aug_in = None
        self.pp_dyn_Kaug = None
        self.pp_dyn_Kauginv = None
        self.pp_dyn_psi1 = None
        self.pp_dyn_psi2 = None

        self.pp_dyn_kernparam_alpha = None
        self.pp_dyn_kernparam_sigmasqrf = None
        self.pp_dyn_kernparam_lambdas = None
        self.pp_dyn_aug_out_mean = None
        self.pp_dyn_aug_out_covar = None

        self.pp_lvm_aug_in = None
        self.pp_lvm_Kaug = None
        self.pp_lvm_Kauginv = None
        self.pp_lvm_psi1 = None
        self.pp_lvm_psi2 = None
        self.pp_lvm_kernparam_beta = None
        self.pp_lvm_kernparam_sigmasqrf = None
        self.pp_lvm_kernparam_lambdas = None
        self.pp_lvm_aug_out_mean = None
        self.pp_lvm_aug_out_covar = None

    def compute_dyn_lvm_vars(self, ns=nt.NumpyLinalg):
        Q = self.params.Q
        x_means = match(self.x_means_var, ns)
        x_covars_diags = ns.exp(match(self.log_x_covars_diags_var, ns))
        
        # Dynamics
        dyn_kernparam_sigmasqrf = ns.exp(match(self.dyn_log_kernparam_sigmasqrf_var, ns))
        dyn_kernparam_lambda1 = ns.exp(match(self.dyn_log_kernparam_lambda1_var, ns))
        dyn_kernparam_lambda2 = ns.exp(match(self.dyn_log_kernparam_lambda2_var, ns))
        dyn_kernparam_lambdas = ns.concatenate([ns.zeros(Q) + dyn_kernparam_lambda2, 
                                                ns.zeros(Q) + dyn_kernparam_lambda1])
        dyn_xtminus_means = ns.concatenate([x_means[match(self.params.xtminusindexes_vars[0], ns), :], 
                                            x_means[match(self.params.xtminusindexes_vars[1], ns), :]], axis=1)
        dyn_xtminus_covars = ns.concatenate([x_covars_diags[match(self.params.xtminusindexes_vars[0], ns), :], 
                                             x_covars_diags[match(self.params.xtminusindexes_vars[1], ns), :]], axis=1)
        dyn_xt_means = x_means[match(self.params.xtplusindexes_var, ns), :]
        dyn_xt_covars = x_covars_diags[match(self.params.xtplusindexes_var, ns), :]
        dyn_aug_in = match(self.dyn_aug_in_var, ns)
        dyn_psi0 = self.kern_type.psi_stat_0(dyn_kernparam_sigmasqrf, dyn_xtminus_means, ns=ns)
        dyn_Kaug_partial = self.kern_type.gram_matrix(dyn_kernparam_sigmasqrf, 
                                                      dyn_kernparam_lambdas, 
                                                      dyn_aug_in, 
                                                      dyn_aug_in,
                                                      ns=ns)
        dyn_psi1_partial = self.kern_type.psi_stat_1(dyn_kernparam_sigmasqrf, 
                                                     dyn_kernparam_lambdas, 
                                                     dyn_aug_in,
                                                     dyn_xtminus_means,
                                                     dyn_xtminus_covars,
                                                     ns=ns)
        dyn_psi2_partial = self.kern_type.psi_stat_2(dyn_kernparam_sigmasqrf, 
                                                     dyn_kernparam_lambdas, 
                                                     dyn_aug_in,
                                                     dyn_xtminus_means,
                                                     dyn_xtminus_covars,
                                                     ns=ns)

        #psi1_0 = self.kern_type.psi_stat_1_Lawrence(dyn_kernparam_sigmasqrf, 
        #                                            dyn_kernparam_lambdas, 
        #                                            dyn_aug_in,
        #                                            dyn_xtminus_means,
        #                                            dyn_xtminus_covars)

        #psi2_0 = self.kern_type.psi_stat_2_Lawrence(dyn_kernparam_sigmasqrf, 
        #                                             dyn_kernparam_lambdas, 
        #                                             dyn_aug_in,
        #                                             dyn_xtminus_means,
        #                                             dyn_xtminus_covars)

        # (GP) Latent Variable Model
        lvm_aug_in = match(self.lvm_aug_in_var, ns)
        lvm_kernparam_beta = ns.exp(match(self.lvm_log_kernparam_beta_var , ns))
        lvm_kernparam_sigmasqrf = ns.exp(match(self.lvm_log_kernparam_sigmasqrf_var, ns))
        lvm_kernparam_lambda = ns.exp(match(self.lvm_log_kernparam_lambda_var, ns))
        lvm_kernparam_lambdas = np.ones(Q) * lvm_kernparam_lambda
        lvm_Kaug = self.kern_type.gram_matrix(lvm_kernparam_sigmasqrf, 
                                              lvm_kernparam_lambdas, 
                                              lvm_aug_in, 
                                              lvm_aug_in,
                                              ns=ns)
        lvm_psi0 = self.kern_type.psi_stat_0(lvm_kernparam_sigmasqrf, 
                                             x_means,
                                             ns=ns)
        lvm_psi1 = self.kern_type.psi_stat_1(lvm_kernparam_sigmasqrf, 
                                             lvm_kernparam_lambdas, 
                                             lvm_aug_in,
                                             x_means,
                                             x_covars_diags,
                                             ns=ns)
        lvm_psi2 = self.kern_type.psi_stat_2(lvm_kernparam_sigmasqrf, 
                                             lvm_kernparam_lambdas, 
                                             lvm_aug_in,
                                             x_means,
                                             x_covars_diags,
                                             ns=ns)
        return [x_means,
                x_covars_diags,
                dyn_xt_means,
                dyn_xt_covars,
                dyn_Kaug_partial, 
                dyn_psi0,
                dyn_psi1_partial,
                dyn_psi2_partial,
                lvm_kernparam_beta,
                lvm_Kaug,
                lvm_psi0,
                lvm_psi1,
                lvm_psi2]

    def precalc_posterior_predictive_psi(self):
        [self.pp_x_means,
         x_covars_diags,
         self.pp_dyn_xt_means,
         self.pp_dyn_xt_covars,
         self.pp_dyn_Kaug_partial, 
         dyn_psi0,
         self.pp_dyn_psi1_partial,
         self.pp_dyn_psi2_partial,
         lvm_kernparam_beta,
         self.pp_lvm_Kaug,
         lvm_psi0,
         self.pp_lvm_psi1,
         self.pp_lvm_psi2] = self.compute_dyn_lvm_vars(ns=nt.NumpyLinalg)

        Q = self.params.Q
        self.pp_dyn_kernparam_sigmasqrf = np.exp(self.dyn_log_kernparam_sigmasqrf_var.val)
        self.pp_dyn_kernparam_lambdas = np.concatenate([np.zeros(Q) + np.exp(self.dyn_log_kernparam_lambda2_var.val), 
                                                        np.zeros(Q) + np.exp(self.dyn_log_kernparam_lambda1_var.val)])
        self.pp_dyn_aug_in = self.dyn_aug_in_var.val

        self.pp_lvm_aug_in = self.lvm_aug_in_var.val
        self.pp_lvm_Kauginv = np.linalg.inv(self.pp_lvm_Kaug)
        self.pp_lvm_kernparam_beta = np.exp(self.lvm_log_kernparam_beta_var.val)
        self.pp_lvm_kernparam_sigmasqrf = np.exp(self.lvm_log_kernparam_sigmasqrf_var.val)
        self.pp_lvm_kernparam_lambdas = np.ones(Q) * np.exp(self.lvm_log_kernparam_lambda_var.val)

        # Optimal dynamics U
        self.pp_dyn_aug_out_mean = None
        self.pp_dyn_aug_out_covar = None

        # Optimal LVM U
        self.pp_lvm_aug_out_mean = None
        self.pp_lvm_aug_out_covar = None

    def precalc_posterior_predictive_aug_out(self):
        # Optimal dynamics U
        self.pp_dyn_aug_out_mean, self.pp_dyn_aug_out_covar = vceq.optimal_q_u(self.pp_dyn_Kauginv, 
                                                                               self.pp_dyn_kernparam_alpha, 
                                                                               self.pp_dyn_xt_means, 
                                                                               self.pp_dyn_psi1, 
                                                                               self.pp_dyn_psi2)

        # Optimal LVM U
        self.pp_lvm_aug_out_mean, self.pp_lvm_aug_out_covar = vceq.optimal_q_u(self.pp_lvm_Kauginv, 
                                                                               1.0 / self.pp_lvm_kernparam_beta, 
                                                                               self.params.data.Y_centered_value, 
                                                                               self.pp_lvm_psi1, 
                                                                               self.pp_lvm_psi2)
        pass
        
    def posterior_predictive_dyn_kern_partial(self, x1, x2):
        res = self.kern_type.gram_matrix(self.pp_dyn_kernparam_sigmasqrf, 
                                         self.pp_dyn_kernparam_lambdas, 
                                         x1, 
                                         x2,
                                         ns=nt.NumpyLinalg)
        return res

    def posterior_predictive_lvm_kern(self, x1, x2):
        res = self.kern_type.gram_matrix(self.pp_lvm_kernparam_sigmasqrf, 
                                         self.pp_lvm_kernparam_lambdas, 
                                         x1, 
                                         x2,
                                         ns=nt.NumpyLinalg)
        return res

    def posterior_predictive_lvm_ystar(self, xstar):
        K_xstar_aug_in = self.posterior_predictive_lvm_kern(xstar, self.pp_lvm_aug_in)
        K_xstar_xstar = self.posterior_predictive_lvm_kern(xstar, xstar)

        Astar = np.dot(K_xstar_aug_in, self.pp_lvm_Kauginv)
        ystar_mean = np.dot(Astar, self.pp_lvm_aug_out_mean) + self.params.data.Y_mean_value
        ystar_covar = K_xstar_xstar \
                      - np.dot(Astar, K_xstar_aug_in.T) \
                      + np.dot(Astar, np.dot(self.pp_lvm_aug_out_covar, Astar.T))
        return ystar_mean, ystar_covar

    def posterior_predictive_dyn_xtstar(self, K_xminuststar_aug_in, K_xminuststar_xminuststar):
        Astar = np.dot(K_xminuststar_aug_in, self.pp_dyn_Kauginv)
        xtstar_mean = np.dot(Astar, self.pp_dyn_aug_out_mean)
        xtstar_covar = K_xminuststar_xminuststar \
                      - np.dot(Astar, K_xminuststar_aug_in.T) \
                      + np.dot(Astar, np.dot(self.pp_dyn_aug_out_covar, Astar.T))
        return xtstar_mean, xtstar_covar

    def elbo(self):
        ns = self.ns
        dyn_elbo = vceq.VariationalGPDM.elbo_diag_x_covars(
                                                  self.dyn_xt_means, 
                                                  self.dyn_xt_covars, 
                                                  self.dyn_kernparam_alpha,
                                                  self.dyn_Kaug,
                                                  self.dyn_psi0,
                                                  self.dyn_psi1,
                                                  self.dyn_psi2,
                                                  ns)
        lvm_elbo = vceq.VariationalGPLVM.elbo(self.params.data.Y_centered, 
                                              self.lvm_kernparam_beta, 
                                              self.lvm_Kaug, 
                                              self.lvm_psi0,
                                              self.lvm_psi1,
                                              self.lvm_psi2,
                                              ns)
        
        #x0_elbo = vceq.gaussians_cross_entropies(self.x_means[self.params.xt0indexes, :],
        #                                         self.x_covars[self.params.xt0indexes, :, :],
        #                                         ns.zeros_like(self.x_means[self.params.xt0indexes, :]),
        #                                         ns.identity(self.params.Q)[ns.newaxis, :, :] + ns.zeros(self.params.xt0indexes_var.val.size)[:, ns.newaxis, ns.newaxis],
        #                                         self.ns)
        #x0_elbo += vceq.gaussians_cross_entropies(self.x_means[self.params.xt0indexes+1, :], 
        #                                          self.x_covars[self.params.xt0indexes+1, :, :], 
        #                                          self.x_means[self.params.xt0indexes, :], 
        #                                          ns.identity(self.params.Q)[ns.newaxis, :, :] + ns.zeros(self.params.xt0indexes_var.val.size)[:, ns.newaxis, ns.newaxis],
        #                                          self.ns)
        #x0_elbo = self.ns.sum(x0_elbo)
        
        x0_elbo = 0
        for x0index in self.params.xt0indexes_var.val:
            x0_elbo += vceq.gaussians_cross_entropy(self.x_means[x0index, :], ns.diag(self.x_covars_diags[x0index, :]), 
                                                    ns.zeros(self.params.Q), ns.identity(self.params.Q), self.ns)
            x0_elbo += vceq.gaussians_cross_entropy(self.x_means[x0index+1, :], ns.diag(self.x_covars_diags[x0index+1, :]), 
                                                    self.x_means[x0index, :], ns.identity(self.params.Q), self.ns)
        #return dyn_elbo
        return dyn_elbo + lvm_elbo + x0_elbo

def append_np_lists(a, b):
    if not a:
        return b
    if not b:
        return a
    res = []
    for i in range(len(a)):
        res.append(np.concatenate((a[i], b[i]), axis=0))
    return res


class VCGPDM(pf.ParametricFunction):
    def __init__(self, 
                 modelparams,  # =ModelParams()
                 ns=nt.NumpyLinalg):
        super(VCGPDM, self).__init__()
        self.ns = ns
        self.params = modelparams
        W = self.params.nparts
        self.parts = [VCGPDMPart(self.params.make_part_params(part_ID, ns), ns) for part_ID in range(self.params.nparts)]

        coupling_diag = 0.01
        coupling_cross = 10.1
        couplingcovars = coupling_cross + np.zeros([W, W]) - (coupling_cross - coupling_diag) * np.identity(W)
        self.log_couplingcovars_var = pf.MatrixVariable("log_couplingcovars", np.log(couplingcovars), bounds=(-5.0, 5.0))  # [W*W], from i-th part to j-th 
        self.couplingcovars = ns.exp(match(self.log_couplingcovars_var, ns))

        self.noisecovars = 1.0 / ns.sum(1.0 / self.couplingcovars, axis=0)  # [W], total covariance for i-th part dynamics
        self.kernelweights = self.noisecovars[np.newaxis, :]**2 / self.couplingcovars**2

        # Compute the coupled dynamics augmenting kernel matrices 
        for j in range(self.params.nparts):
            part = self.parts[j]
            part.dyn_Kaug = 0
            part.dyn_kernparam_alpha = self.noisecovars[j]
            for i in range(self.params.nparts):
                part.dyn_Kaug += self.kernelweights[i, j] * self.parts[i].dyn_Kaug_partial
            part.dyn_Kauginv = ns.inv(part.dyn_Kaug)

        # Compute Psi_1-statistics for the dynamics
        for j in range(self.params.nparts):
            part = self.parts[j]
            part.dyn_psi1 = 0
            for i in range(self.params.nparts):
                part.dyn_psi1 += self.kernelweights[i, j] * self.parts[i].dyn_psi1_partial
                
        # Compute Psi_2-statistics for the dynamics
        for j in range(self.params.nparts):
            part = self.parts[j]
            part.dyn_psi2 = 0
            for i in range(self.params.nparts):
                for k in range(self.params.nparts):
                    if i == k:
                        part.dyn_psi2 += self.kernelweights[i, j]**2 * self.parts[i].dyn_psi2_partial  # as it is quadratic
                    else:
                        part.dyn_psi2 += self.kernelweights[i, j] * self.parts[i].dyn_psi1_partial[:, np.newaxis, :] \
                                       * self.kernelweights[k, j] * self.parts[k].dyn_psi1_partial[:, :, np.newaxis]
        
        self._consts = [self.params.parts_indexes_var,
                        #self.params.xt0indexes_var,
                        self.params.xtplusindexes_var
                        ] + self.params.xtminusindexes_vars
        #self._consts = [self.params.parts_indexes_var]
        self._params = [self.log_couplingcovars_var] 
        self._args = [self.params.data.Y_var]
        self._children = self.parts
        if ns == nt.NumpyLinalg:
            self.function = self.elbo
            self.symbolic = None
            res = -self.function()
            print("ELBO shape: ", res.shape)
            print("Initial neg-ELBO: ", res)
        elif ns == nt.TheanoLinalg:
            self.function = None
            self.symbolic = self.elbo
            self.functominimize = thx.FuncWithGrad(expr=-self.elbo(), args=pf.symbols(self.get_all_vars()))
            self.functominimize.set_args_values(pf.values(self.get_all_vars()))
            res = self.functominimize.get_func_value()
            print("ELBO shape: ", res.shape)
            print("Initial neg-ELBO: ", res)
        
        # Optimization mode
        self.optimizing_log_x_covars_diag = True  # set to False and use small covariances for MAP-like learning
            
          
    def precalc_posterior_predictive(self):
        # Step 1. Partial Psi-stats
        for part in self.parts:
            part.precalc_posterior_predictive_psi()
        
        # Step 2. Full coupled dynamics augmenting kernel matrices and Psi-stats
        couplingcovars = np.exp(match(self.log_couplingcovars_var, ns=nt.NumpyLinalg))
        noisecovars = 1.0 / np.sum(1.0 / couplingcovars, axis=0)  # [W], total covariance for i-th part dynamics
        kernelweights = noisecovars[np.newaxis, :]**2 / couplingcovars**2
        for j in range(self.params.nparts):
            part = self.parts[j]
            part.pp_dyn_Kaug = 0
            part.pp_dyn_kernparam_alpha = noisecovars[j]
            for i in range(self.params.nparts):
                part.pp_dyn_Kaug += kernelweights[i, j] * self.parts[i].pp_dyn_Kaug_partial
            part.pp_dyn_Kauginv = np.linalg.inv(part.pp_dyn_Kaug)

        # Compute Psi_1-statistics for the dynamics
        for j in range(self.params.nparts):
            part = self.parts[j]
            part.pp_dyn_psi1 = 0
            for i in range(self.params.nparts):
                part.pp_dyn_psi1 += kernelweights[i, j] * self.parts[i].pp_dyn_psi1_partial

        # Compute Psi_2-statistics for the dynamics
        for j in range(self.params.nparts):
            part = self.parts[j]
            part.pp_dyn_psi2 = 0
            for i in range(self.params.nparts):
                for k in range(self.params.nparts):
                    if i == k:
                        part.pp_dyn_psi2 += kernelweights[i, j]**2 * self.parts[i].pp_dyn_psi2_partial  # as it is quadratic
                    else:
                        part.pp_dyn_psi2 += kernelweights[i, j] * self.parts[i].pp_dyn_psi1_partial[:, np.newaxis, :] \
                                          * kernelweights[k, j] * self.parts[k].pp_dyn_psi1_partial[:, :, np.newaxis]

        # Step 3. Optimal p(x_out)
        for part in self.parts:
            part.precalc_posterior_predictive_aug_out()

    def give_first_xminust_stars(self):
        xminust_stars = []
        for i in range(self.params.nparts):
            part = self.parts[i]
            xminust_stars.append(np.concatenate([part.pp_x_means[part.params.xtminusindexes_vars[0].val[0], :], 
                                                 part.pp_x_means[part.params.xtminusindexes_vars[1].val[0], :]], axis=1))
        return xminust_stars
    
    def update_xminust_stars(self, xminust_stars, xtstars):
        new_xminust_stars = []
        for i in range(self.params.nparts):
            part = self.parts[i]
            new_xminust_star = np.concatenate((xminust_stars[i][part.params.Q:], xtstars[i][0]))
            new_xminust_stars.append(new_xminust_star)
        return new_xminust_stars
    
    def run_generative_mode(self, nsteps=100, xminust_stars=None):
        if xminust_stars == None:
            xminust_stars = self.give_first_xminust_stars()
        x_pathes0 = [xminust_stars[i][np.newaxis, :xminust_stars[i].size/2] for i in range(self.params.nparts)]
        x_pathes1 = [xminust_stars[i][np.newaxis, xminust_stars[i].size/2:] for i in range(self.params.nparts)]
        x_pathes = append_np_lists(x_pathes0, x_pathes1)
        y_pathes0, c = self.lvm_posterior_predictive([x_pathes[i][0] for i in range(self.params.nparts)])
        y_pathes1, c = self.lvm_posterior_predictive([x_pathes[i][1] for i in range(self.params.nparts)])
        y_pathes = append_np_lists(y_pathes0, y_pathes1)
        for i in range(nsteps):
            xtstars, c = self.dyn_posterior_predictive(xminust_stars)
            ystars, c = self.lvm_posterior_predictive(xtstars)
            xminust_stars = self.update_xminust_stars(xminust_stars, xtstars)
            x_pathes = append_np_lists(x_pathes, xtstars)
            y_pathes = append_np_lists(y_pathes, ystars)
        return x_pathes, y_pathes

    def dyn_posterior_predictive(self, xminust_stars):
        """
        :param xtminus_stars: list of size W of [2*Q[w]] xminust_star_part for W parts
        :return: list of xt_star for W parts
        """
        couplingcovars = np.exp(match(self.log_couplingcovars_var, ns=nt.NumpyLinalg))
        noisecovars = 1.0 / np.sum(1.0 / couplingcovars, axis=0)  # [W], total covariance for i-th part dynamics
        kernelweights = noisecovars[np.newaxis, :]**2 / couplingcovars**2
        K_xminuststar_aug_in_partial = []
        K_xminuststar_xminuststar_partial = []
        for i in range(self.params.nparts):
            part = self.parts[i]
            xminust_star = xminust_stars[i]
            K_xminuststar_aug_in_partial.append(part.posterior_predictive_dyn_kern_partial(xminust_star, part.pp_dyn_aug_in))
            K_xminuststar_xminuststar_partial.append(part.posterior_predictive_dyn_kern_partial(xminust_star, xminust_star))
        K_xminuststar_aug_in = []
        K_xminuststar_xminuststar = []
        for j in range(self.params.nparts):
            s = 0
            for i in range(self.params.nparts):
                s += kernelweights[i, j] * K_xminuststar_aug_in_partial[i]
            K_xminuststar_aug_in.append(s)
            s = 0
            for i in range(self.params.nparts):
                s += kernelweights[i, j] * K_xminuststar_xminuststar_partial[i]
            K_xminuststar_xminuststar.append(s)
        xt_stars_mean = []
        xt_stars_covar = []
        for i in range(self.params.nparts):
            part = self.parts[i]
            xtstar_mean, xtstar_covar = part.posterior_predictive_dyn_xtstar(K_xminuststar_aug_in[i], K_xminuststar_xminuststar[i])
            xt_stars_mean.append(xtstar_mean)
            xt_stars_covar.append(xtstar_covar)
        return xt_stars_mean, xtstar_covar
        
    def lvm_posterior_predictive(self, x_stars):
        """
        :param x_stars: list of size W of [Q[w]] x_star for W parts
        :return: list of y_stars for W parts
        """
        y_stars_mean = []
        y_stars_covar = []
        for i in range(self.params.nparts):
            part = self.parts[i]
            m, c = part.posterior_predictive_lvm_ystar(x_stars[i])
            y_stars_mean.append(m)
            y_stars_covar.append(c)
        return y_stars_mean, y_stars_covar

    def elbo(self):
        res = 0
        for part in self.parts:
            res += part.elbo()
        return res

    def get_coupling_matrix_vales(self):
        return np.exp(self.log_couplingcovars_var.val)

    def optimize_all(self, maxiter=10):
        print("|================================================================|")
        print("Optimizing all variational parameters")
        self.functominimize.set_args_values(pf.values(self.get_all_vars()))
        params_to_minimize = self.get_params()
        pfm.l_bfgs_b(self.functominimize, params_to_minimize, maxiter=maxiter)
        
    def optimize_x(self, maxiter=10):
        print("|================================================================|")
        print("Optimizing latent X")
        self.functominimize.set_args_values(pf.values(self.get_all_vars()))
        params_to_minimize = []
        for part in self.parts:
            params_to_minimize += [part.x_means_var] 
            if self.optimizing_log_x_covars_diag:
                params_to_minimize += [part.log_x_covars_diags_var]
        pfm.l_bfgs_b(self.functominimize, params_to_minimize, maxiter=maxiter)

    def optimize_x_and_inducing(self, maxiter=10):
        print("|================================================================|")
        print("Optimizing latent X and inducing inputs")
        self.functominimize.set_args_values(pf.values(self.get_all_vars()))
        params_to_minimize = []
        for part in self.parts:
            params_to_minimize += [part.x_means_var, 
                                   part.dyn_aug_in_var, 
                                   part.lvm_aug_in_var]
            if self.optimizing_log_x_covars_diag:
                params_to_minimize += [part.log_x_covars_diags_var]
        pfm.l_bfgs_b(self.functominimize, params_to_minimize, maxiter=maxiter)

    def optimize_inducing(self, maxiter=10):
        print("|================================================================|")
        print("Optimizing inducing inputs")
        self.functominimize.set_args_values(pf.values(self.get_all_vars()))
        params_to_minimize = []
        for part in self.parts:
            params_to_minimize += [part.dyn_aug_in_var, 
                                   part.lvm_aug_in_var]
        pfm.l_bfgs_b(self.functominimize, params_to_minimize, maxiter=maxiter)
        
    def optimize_kernel_params(self, maxiter=10):
        print("|================================================================|")
        print("Optimizing kernel parameters")
        self.functominimize.set_args_values(pf.values(self.get_all_vars()))
        params_to_minimize = [self.log_couplingcovars_var]
        for part in self.parts:
            params_to_minimize += [part.dyn_log_kernparam_sigmasqrf_var,
                                   part.dyn_log_kernparam_lambda1_var,
                                   part.dyn_log_kernparam_lambda2_var,
                                   part.lvm_log_kernparam_beta_var,
                                   part.lvm_log_kernparam_sigmasqrf_var,
                                   part.lvm_log_kernparam_lambda_var]
        pfm.l_bfgs_b(self.functominimize, params_to_minimize, maxiter=maxiter)
    
    def fix_x_covars_diags(self, covar=None):
        self.optimizing_log_x_covars_diag = False
        return self.set_x_covars_diags(covar)

    def set_x_covars_diags(self, covar=None):
        if covar is None:
            covar = 1e-9
        for part in self.parts:
            part.log_x_covars_diags_var.val = np.log(covar + 0 * part.log_x_covars_diags_var.val)
        self.functominimize.set_args_values(pf.values(self.get_all_vars()))
        return self.functominimize.get_func_value()

    def elbo_wrt_x_covars(self, covars=None):
        if covars is None:
            covars = np.exp(np.linspace(np.log(1e-90), np.log(1.0), 200))
        stored_covars = [np.copy(part.log_x_covars_diags_var.val) for part in self.parts]
        res = [self.set_x_covars_diags(covar) for covar in covars]
        for part, covar in zip(self.parts, stored_covars):
            part.log_x_covars_diags_var.val = covar
        return res, covars

    def print_min_max_x_covars_diags(self):
        print("Min and Max of x_covars_diags: ") 
        for part in self.parts:
            print("min(x_covars_diags) = ", np.min(np.exp(part.log_x_covars_diags_var.val)))
            print("max(x_covars_diags) = ", np.max(np.exp(part.log_x_covars_diags_var.val)))

def plot_elbo_wrt_x_covars(model, covars=None):
    elbos, covars = model.elbo_wrt_x_covars(covars)
    plt.figure()
    plt.title("Neg-ELBO wrt x_covars")
    plt.plot(covars, elbos)
    plt.show()

def plot_model(model, nsteps=None, path_to_save=None):
    if path_to_save is not None:
        save_plots(model, nsteps, path_to_save)

    print("Coupling matrix: ")
    print(model.get_coupling_matrix_vales())
    if nsteps is None:
        nsteps = model.params.data.N
    model.precalc_posterior_predictive()
    x_pathes, y_pathes = model.run_generative_mode(nsteps)
    nparts = len(model.parts)
    ncols = 3
    plt.figure()
    for i in range(nparts):
        part = model.parts[i]
        
        plt.subplot(nparts, ncols, 1+i*ncols)
        plt.title("Latent space " + str(i))
        plt.plot(part.pp_x_means[:, 0], part.pp_x_means[:, 1], '-', alpha=0.2)
        plt.plot(part.pp_dyn_aug_in[:, 0], part.pp_dyn_aug_in[:, 1], 'o', markersize=15, markeredgewidth=2, fillstyle="none")
        plt.plot(part.pp_lvm_aug_in[:, 0], part.pp_lvm_aug_in[:, 1], '+', markersize=15, markeredgewidth=2, fillstyle="none")
        
        plt.subplot(nparts, ncols, 2+i*ncols)
        plt.title("Latent space trajectory " + str(i))
        plt.plot(part.pp_x_means[:, 0], part.pp_x_means[:, 1], '-', alpha=0.2)
        plt.plot(x_pathes[i][:, 0], x_pathes[i][:, 1], color="k")

        plt.subplot(nparts, ncols, 3+i*ncols)
        plt.title("Observed space trajectory " + str(i))
        plt.plot(part.params.data.Y_value, alpha=0.2)
        plt.plot(y_pathes[i])
    plt.show()

def save_plots(model, nsteps=None, directory=None):
    if directory is None:
        return
    if not os.path.exists(directory):
        os.makedirs(directory)
    if nsteps is None:
        nsteps = model.params.data.N
    model.precalc_posterior_predictive()
    x_pathes, y_pathes = model.run_generative_mode(nsteps)
    nparts = len(model.parts)
    for i in range(nparts):
        part = model.parts[i]
        
        fig = plt.figure(figsize=(5, 5))
        plt.plot(part.pp_x_means[:, 0], part.pp_x_means[:, 1], '-', alpha=0.2)
        plt.plot(part.pp_dyn_aug_in[:, 0], part.pp_dyn_aug_in[:, 1], 'o', markersize=15, markeredgewidth=2, color="k", fillstyle="none")
        plt.plot(part.pp_lvm_aug_in[:, 0], part.pp_lvm_aug_in[:, 1], '+', markersize=15, markeredgewidth=2, color="k", fillstyle="none")
        plt.savefig("{}/part_{}_latent_points.pdf".format(directory, i))
        plt.close(fig)
        
        fig = plt.figure(figsize=(5, 5))
        plt.plot(part.pp_x_means[:, 0], part.pp_x_means[:, 1], '-', alpha=0.2)
        plt.plot(x_pathes[i][:, 0], x_pathes[i][:, 1], color="k")
        plt.savefig("{}/part_{}_latent_trajectory.pdf".format(directory, i))
        plt.close(fig)

        fig = plt.figure(figsize=(5, 5))
        plt.plot(part.params.data.Y_value[:nsteps, :], alpha=0.2)
        plt.gca().set_color_cycle(None)
        plt.plot(y_pathes[i])
        plt.savefig("{}/part_{}_observed_trajectory.pdf".format(directory, i))
        plt.close(fig)


def variance_explained(train, gen):
    assert train.shape == gen.shape
    train_mean = np.mean(train, axis=0)
    train_centered = train - train_mean

    gen_mean = np.mean(gen, axis=0)
    gen_centered = gen - gen_mean

    var_err = np.sum((train_centered - gen_centered)**2)
    var_train = np.sum(train_centered**2) 
    var_unexplained = var_err / var_train
    var_explained = 1.0 - var_unexplained
    return var_explained


def variance_explained_training(model):
    nsteps = model.params.data.N
    nsteps = 200
    model.precalc_posterior_predictive()
    x_pathes, y_pathes = model.run_generative_mode(nsteps-2)
    nparts = len(model.parts)
    res = []
    for i in range(nparts):
        part = model.parts[i]
        var_explained = variance_explained(train=part.params.data.Y_value[:nsteps, :], gen=y_pathes[i])
        res.append(var_explained)
    return res

def variance_explained_generated(model, after_training_data):
    nsteps = model.params.data.N
    nsteps = 200
    model.precalc_posterior_predictive()
    xminust_stars = []
    for i in range(model.params.nparts):
        part = model.parts[i]
        xminust_stars.append(np.concatenate([part.pp_x_means[-2, :], 
                                             part.pp_x_means[-1, :]], axis=1))
    
    x_pathes, y_pathes = model.run_generative_mode(nsteps, xminust_stars)
    nparts = len(model.parts)
    res = []
    for i in range(nparts):
        part = model.parts[i]
        var_explained = variance_explained(train=after_training_data[:, model.params.parts_indexes_var[i].val], gen = y_pathes[i][2:, :])
        res.append(var_explained)
    return res


