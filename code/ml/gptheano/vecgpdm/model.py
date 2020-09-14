import os
import time
import logging
from six.moves import cPickle
import numpy as np
import scipy as sp
import numerical.numpytheano as nt
import numerical.numpyext.linalg as ntla
import numerical.numpyext as npx
import numerical.numpyext.dimreduction as dr
import theano
import numerical.numpytheano.theanopool as tp
import numerical.numpytheano.optimization as opt
import ml.gptheano.vecgpdm.equations as vceq
import ml.gptheano.vecgpdm.kernels as krn
import ml.gptheano.vecgpdm.gaussianprocess as gpr
import ml.gp.equations as gpeq
import numerical.theanoext as thx
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlibex.mlplot as plx
from ml.gptheano.vecgpdm.equations import list_of_nones
from ml.gptheano.vecgpdm.equations import matrix_of_nones
from ml.gptheano.vecgpdm.enums import *
from ml.gptheano.vecgpdm.modelplots import *

"""
Model of an explicitly coupled dynamics WITHOUT exchangeable primitives.
It is the original unmarginalized version of CGPDM.
"""

pl = logging.getLogger(__name__)

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

class PartParam(object):
    def __init__(self,
                 data,  # =PartData(),
                 Q,  # =2, latent dimensionality
                 xt0indexes_var,
                 xtminusindexes_vars,
                 xtplusindexes_var,
                 xt0indexes,
                 xtminusindexes,
                 xtplusindexes,
                 dyn_M,  # =10, number of inducing mappings
                 lvm_M,
                 ID,
                 suffix,  # string description
                 init_func
                 ):
        self.data = data
        self.Q = Q
        self.dynamics_order = 2
        self.xt0indexes_var, self.xtminusindexes_vars, self.xtplusindexes_var = xt0indexes_var, xtminusindexes_vars, xtplusindexes_var
        self.xt0indexes, self.xtminusindexes, self.xtplusindexes = xt0indexes, xtminusindexes, xtplusindexes
        self.dyn_M = dyn_M
        self.lvm_M = lvm_M
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
        self.Y = ns.matrix("Y", np.vstack(Y_sequences))
        ###self.Y_var = pf.MatrixVariable("Y", np.vstack(Y_sequences))
        ###self.Y = match(self.Y_var, ns)
        self.sequences_indexes = gpeq.sequences_indexes(Y_sequences)
        self.N, self.D = ns.get_value(self.Y).shape


class ModelParam(object):
    def __init__(self,
                 data,  # =ModelData(),
                 Qs,  # =[2, 2, 2], latent dimensions
                 parts_IDs,  # =[0, 0, 1, 1, 2, 2]
                 dyn_Ms,  # =[8, 8, 8], number of inducing mappings for dynamics
                 lvm_Ms,  # =[8, 8, 8], number of inducing mappings for the X->Y mapping
                 init_func=None,
                 dyn_kern_type=krn.RBF_Kernel_noscale,
                 dyn_kern_boost=False,
                 lvm_kern_type=krn.RBF_Kernel,
                 lvm_kern_boost=False,
                 estimation_mode=EstimationMode.ELBO,
                 ns=nt.NumpyLinalg):
        self.data = data
        self.Qs = Qs
        self.parts_IDs = parts_IDs
        self.dynamics_order = 2
        self.dyn_Ms = list(dyn_Ms)
        self.lvm_Ms = list(lvm_Ms)
        self.init_func = init_func
        self.dyn_kern_type = dyn_kern_type
        self.dyn_kern_boost = dyn_kern_boost
        self.lvm_kern_type = lvm_kern_type
        self.lvm_kern_boost = lvm_kern_boost
        self.estimation_mode = estimation_mode
                 
        assert len(parts_IDs) == data.D
        assert min(parts_IDs) == 0
        self.nparts = max(parts_IDs) + 1
        self.parts_indexes = [ns.int_vector("parts_indexes_[{}]".format(i), np.array([j for j, index in enumerate(parts_IDs) if i == index])) for i in range(self.nparts)]
        
        xt0indexes, xtminusindexes, xtplusindexes = gpeq.xt0_xtminus_xtplus_indexes(data.sequences_indexes, self.dynamics_order)
        self.xt0indexes = ns.int_vector("xt0indexes", xt0indexes)
        self.xtminusindexes = [ns.int_vector("xtminusindexes_[{}]".format(i), indexes) for i, indexes in enumerate(xtminusindexes)]
        self.xtplusindexes = ns.int_vector("xtplusindexes", xtplusindexes)
        self.parts = [self._make_part_params(part_ID, ns=ns) for part_ID in range(self.nparts)]

    def set_Ms(self, dyn_Ms, lvm_Ms):
        self.dyn_Ms = list(dyn_Ms)
        self.lvm_Ms = list(lvm_Ms)
        for i, part in enumerate(self.parts):
            part.dyn_M = dyn_Ms[i]
            part.lvm_M = lvm_Ms[i]

    def _make_part_params(self, part_ID, ns=nt.NumpyLinalg):
        res = PartParam(PartData(ns.get_value(self.data.Y)[:, ns.get_value(self.parts_indexes[part_ID])], 
                                  self.data.Y[:, ns.get_value(self.parts_indexes[part_ID])], 
                                  ns),
                         self.Qs[part_ID],
                         self.xt0indexes,
                         self.xtminusindexes,
                         self.xtplusindexes,
                         self.xt0indexes,
                         self.xtminusindexes,
                         self.xtplusindexes,
                         self.dyn_Ms[part_ID],
                         self.lvm_Ms[part_ID],
                         part_ID,
                         "_" + str(part_ID),
                         self.init_func)
        return res


class VECGPDM(object):
    def __init__(self, 
                 param,  # =ModelParam()
                 ns=nt.NumpyLinalg):
        super(VECGPDM, self).__init__()
        #np.random.seed(12345)
        self.ns = ns
        self.param = param
        W = self.param.nparts  # number of parts
      
        nmappings = ns.get_value(self.param.xtplusindexes).size
        
        # Coupling uncertainty
        coupling_diag = 1.0 
        coupling_cross = 2.0
        couplingcovars = coupling_cross + np.zeros([W, W]) - (coupling_cross - coupling_diag) * np.identity(W)
        self.alpha = ns.matrix("alpha", couplingcovars, bounds=(1.0e-3, 10.0), tags=(VarTag.couplings))

        # Initialize latent X
        self.x_means = list_of_nones(W)
        self.x_covars_diags = list_of_nones(W)
        self.dyn_xtminus_means = list_of_nones(W)
        self.dyn_xtminus_covars = list_of_nones(W)
        self.dyn_xt_means = list_of_nones(W)
        self.dyn_xt_covars = list_of_nones(W)
        step = list_of_nones(W) 
        for i in range(W):
            suffix = "_[{}]".format(i)
            if self.param.init_func is None:
                init_x = np.sqrt(self.param.parts[i].data.Y_centered_value.shape[0]) * dr.PCAreduction(self.param.parts[i].data.Y_centered_value, self.param.parts[i].Q)
            else:
                init_x = self.param.init_func(self.param.parts[i].data.Y_centered_value, self.param.parts[i].Q, self.param.ID)
            self.x_means[i] = ns.matrix("x_means"+suffix, init_x, bounds=(-10.0, 10.0), tags=(VarTag.latent_x))
            step[i] = ns.get_value(self.x_means[i])[ns.get_value(self.param.xtminusindexes[1]), :] - ns.get_value(self.x_means[i])[ns.get_value(self.param.xtminusindexes[0]), :]
            step[i] = np.mean(np.sqrt(step[i] * step[i]))
            self.x_covars_diags[i] = ns.matrix("x_covars_diags"+suffix, 0.1 * step[i] + np.zeros_like(init_x), bounds=(1.0e-3, 2.0), tags=(VarTag.latent_x))
            self.dyn_xtminus_means[i] = ns.concatenate([self.x_means[i][self.param.xtminusindexes[0], :], 
                                                        self.x_means[i][self.param.xtminusindexes[1], :]], axis=1)
            self.dyn_xtminus_covars[i] = ns.concatenate([self.x_covars_diags[i][self.param.xtminusindexes[0], :], 
                                                         self.x_covars_diags[i][self.param.xtminusindexes[1], :]], axis=1)
            self.dyn_xt_means[i] = self.x_means[i][self.param.xtplusindexes, :]
            self.dyn_xt_covars[i] = self.x_covars_diags[i][self.param.xtplusindexes, :]
            
        # Equations chunks for posterior predictive of dynamics. For q(u*) computation
        self.pp_dyn_FF_eq = list_of_nones(W)
        self.pp_dyn_GG_eq = list_of_nones(W)
        self.pp_dyn_Kzzdiag_eq = list_of_nones(W)
        
        # Optimal q(u) for posterior predictive
        self.pp_dyn_aug_z = matrix_of_nones(W, W)
        self.pp_dyn_aug_u_means = matrix_of_nones(W, W)
        self.pp_dyn_aug_u_covars = matrix_of_nones(W, W)
        
        # Gram matrices of augmening inputs
        self.pp_dyn_Kauginv = matrix_of_nones(W, W)
        self.pp_alpha = None
        
        # Construct dynamics ELBO
        self.dyn_kernel_objs = matrix_of_nones(W, W)  # kernel objects
        self.dyn_aug_z = matrix_of_nones(W, W)  # list of augmenting (inducing) inputs for all W*W parts partial cross-mappings
        self.dyn_Kaug = matrix_of_nones(W, W)  # kernel matrices of augmenting inputs
        self.dyn_psi0 = matrix_of_nones(W, W) 
        self.dyn_psi1 = matrix_of_nones(W, W)
        self.dyn_psi2 = matrix_of_nones(W, W)
        for i in range(W):
            for j in range(W):
                suffix = "_[{}, {}]".format(j, i)
                # Create a variable for augmenting imputs
                self.dyn_aug_z[j][i] = ns.matrix("dyn_aug_z"+suffix, 
                        np.zeros([2 * self.param.parts[j].Q, 2 * self.param.parts[j].Q]), 
                        bounds=(-10.0, 10.0), tags=(VarTag.augmenting_inputs))

                kern_width = 1.0
                self.dyn_kernel_objs[j][i] = self.param.dyn_kern_type(2 * self.param.parts[j].Q, kern_width, suffix, ns=ns)
                self.dyn_kernel_objs[j][i].use_diagonal_boost = self.param.dyn_kern_boost
                self.dyn_Kaug[j][i] = self.dyn_kernel_objs[j][i].gram_matrix(self.dyn_aug_z[j][i], self.dyn_aug_z[j][i], ns=ns)
                self.dyn_psi0[j][i] = self.dyn_kernel_objs[j][i].psi_stat_0(self.dyn_xtminus_means[j], self.dyn_xtminus_covars[j])
                self.dyn_psi1[j][i] = self.dyn_kernel_objs[j][i].psi_stat_1(self.dyn_aug_z[j][i], self.dyn_xtminus_means[j], self.dyn_xtminus_covars[j])
                self.dyn_psi2[j][i] = self.dyn_kernel_objs[j][i].psi_stat_2(self.dyn_aug_z[j][i], self.dyn_xtminus_means[j], self.dyn_xtminus_covars[j])

        self.dyn_elbo, (self.pp_dyn_FF_eq, self.pp_dyn_GG_eq, self.pp_dyn_Kzzdiag_eq) = vceq.GPDM.variational_explicit_elbo_diag_x_covars(
            self.dyn_xt_means, 
            self.dyn_xt_covars, 
            self.alpha,
            self.dyn_Kaug,
            self.dyn_psi0,
            self.dyn_psi1,
            self.dyn_psi2,
            ns=ns)
        
        # Variables for LVM "posterior predictive"
        self.pp_lvm_aug_z = list_of_nones(W)
        self.pp_lvm_aug_u_means = list_of_nones(W)
        self.pp_lvm_aug_u_covars = list_of_nones(W)
        self.pp_lvm_Kauginv = list_of_nones(W)
        self.pp_lvm_beta = list_of_nones(W)

        # Construct LVM ELBO
        self.lvm_elbo = 0.0
        self.lvm_log_beta = list_of_nones(W)
        self.lvm_beta = list_of_nones(W)
        self.lvm_kernel_objs = list_of_nones(W)  # kernel objects
        self.lvm_aug_z = list_of_nones(W)  # list of augmenting (inducing) inputs for all W parts
        self.lvm_Kaug = list_of_nones(W)  # kernel matrices of augmenting inputs
        self.lvm_psi0 = list_of_nones(W) 
        self.lvm_psi1 = list_of_nones(W)
        self.lvm_psi2 = list_of_nones(W)
        for i in range(W):
            suffix = "_[{}]".format(i)
            # Create a variable for augmenting imputs
            self.lvm_aug_z[i] = ns.matrix("lvm_aug_z"+suffix, 
                np.zeros([self.param.parts[j].Q, self.param.parts[j].Q]),
                bounds=(-10.0, 10.0), 
                tags=(VarTag.augmenting_inputs))
            kern_width = 1.0
            self.lvm_beta[i] = ns.scalar("lvm_beta"+suffix, 1.0, bounds=(1.0e-2, 1.0e4), tags=(VarTag.kernel_params))  # Precision !!! INVERSE !!! of the x->y mapping noise
            self.lvm_kernel_objs[i] = self.param.lvm_kern_type(self.param.parts[i].Q, kern_width, suffix, ns=ns)
            self.lvm_kernel_objs[i].use_diagonal_boost = self.param.lvm_kern_boost
            self.lvm_Kaug[i] = self.lvm_kernel_objs[i].gram_matrix(self.lvm_aug_z[i], self.lvm_aug_z[i], ns=ns)
            self.lvm_psi0[i] = self.lvm_kernel_objs[i].psi_stat_0(self.x_means[i], self.x_covars_diags[i])
            self.lvm_psi1[i] = self.lvm_kernel_objs[i].psi_stat_1(self.lvm_aug_z[i], self.x_means[i], self.x_covars_diags[i])
            self.lvm_psi2[i] = self.lvm_kernel_objs[i].psi_stat_2(self.lvm_aug_z[i], self.x_means[i], self.x_covars_diags[i])
            self.lvm_elbo += vceq.GPLVM.variational_elbo(self.param.parts[i].data.Y_centered, 
                                              self.lvm_beta[i],
                                              self.lvm_Kaug[i], 
                                              self.lvm_psi0[i],
                                              self.lvm_psi1[i],
                                              self.lvm_psi2[i],
                                              ns=ns)
        self.init_aug_z()
        x0_elbo = 0.0
        for i in range(W):
            for x0index in ns.get_value(self.param.xt0indexes):
                x0_elbo += vceq.gaussians_cross_entropy(self.x_means[i][x0index, :], ns.diag(self.x_covars_diags[i][x0index, :]), 
                                                        ns.zeros(self.param.parts[i].Q), ns.identity(self.param.parts[i].Q), self.ns)
                x0_elbo += vceq.gaussians_cross_entropy(self.x_means[i][x0index+1, :], ns.diag(self.x_covars_diags[i][x0index+1, :]), 
                                                        self.x_means[i][x0index, :], ns.identity(self.param.parts[i].Q), self.ns)

        self.elbo = self.dyn_elbo + self.lvm_elbo + x0_elbo
        self.neg_elbo = -self.elbo
        pl.info("Initial ELBO = {}".format(self.get_elbo_value()))
        #print("Initial coupling matrix :")
        #self.print_by_tags(set([VarTag.couplings]))
        
        # Construct the log-likelihood equation.
        self.loglikelihood = 0
        self.dyn_Kxxs = matrix_of_nones(W, W)
        for i in range(W):
            for j in range(W):
                self.dyn_Kxxs[j][i] = self.dyn_kernel_objs[j][i].gram_matrix(self.dyn_xtminus_means[j], self.dyn_xtminus_means[j], ns=ns)
        #print(ns)
        self.loglikelihood += vceq.GPDM.loglikelihood_coupled_unmarginalised(Kxxs=self.dyn_Kxxs,
                                                                             Xouts=self.dyn_xt_means, 
                                                                             alpha=self.alpha,
                                                                             order=self.param.dynamics_order,
                                                                             ns=ns)
        for i in range(W):
            for x0index in ns.get_value(self.param.xt0indexes):
                self.loglikelihood += vceq.GPDM.loglikelihood_single_x0(X=self.x_means[i][x0index:, :], 
                                                                        order=self.param.dynamics_order,
                                                                        ns=ns)
        self.lvm_Kxxs = list_of_nones(W)
        for i in range(W):
            self.lvm_Kxxs[i] = self.lvm_kernel_objs[i].gram_matrix(self.x_means[i], self.x_means[i], ns=ns)
        for i in range(W):
            self.loglikelihood += vceq.GPLVM.loglikelihood(Kxx=self.lvm_Kxxs[i],
                                                           y=self.param.parts[i].data.Y_centered,
                                                           beta=self.lvm_beta[i],
                                                           ns=ns)
        pl.info("Initial log-likelihood = {}".format(self.get_loglikelihood_value()))
        self.neg_loglikelihood = -self.loglikelihood



    def get_elbo_value(self):
        return self.ns.evaluate(self.elbo)
        
    def get_loglikelihood_value(self):
        return self.ns.evaluate(self.loglikelihood)        

            
    
    def init_aug_z(self, mode="subsample"):
        # Re-initialize the augmenting points

        ns = self.ns
        W = self.param.nparts  # number of parts
        nmappings = self.ns.get_value(self.param.xtplusindexes).size
        
        for i in range(W):
            for j in range(W):
                M = self.param.parts[j].dyn_M                            
                if mode == "subsample":  # Random subsample initialization
                    dyn_xtminus = ns.evaluate(self.dyn_xtminus_means[j])
                    inds = np.concatenate([[0], np.random.choice(a=range(1, len(dyn_xtminus)), size=M-1, replace=False)])
                    dyn_aug_z = dyn_xtminus[inds] 
                elif mode == "imterpolate":  # Uniform interpolated initialization
                    xin0 = ns.get_value(self.x_means[j])[ns.get_value(self.param.xtminusindexes[0])]
                    xin0interp = sp.interpolate.interp1d(np.linspace(0, 1, len(xin0)), xin0, axis=0)(np.linspace(0, 1, M))
                    xin1 = ns.get_value(self.x_means[j])[ns.get_value(self.param.xtminusindexes[1])]
                    xin1interp = sp.interpolate.interp1d(np.linspace(0, 1, len(xin1)), xin1, axis=0)(np.linspace(0, 1, M))
                    dyn_aug_z = np.hstack([xin0interp, xin1interp])                
                self.ns.set_value(self.dyn_aug_z[j][i], dyn_aug_z)

        for i in range(W):
            if mode == "subsample":  # Random subsample initialization
                M = self.param.parts[i].lvm_M
                lvm_xt = ns.get_value(self.x_means[i])
                inds = np.random.choice(a=range(len(lvm_xt)), size=M, replace=False)
                lvm_aug_z = lvm_xt[inds]
            elif mode == "imterpolate":  # Uniform interpolated initialization
                aug_indexes = np.arange(0, nmappings, int(nmappings/self.param.parts[j].lvm_M))[:self.param.parts[j].lvm_M]
                lvm_aug_z = ns.get_value(self.x_means[i])[ns.get_value(self.param.xtminusindexes[0])[aug_indexes], :]
            self.ns.set_value(self.lvm_aug_z[i], lvm_aug_z)

            

    def _optimize(self, maxiter=None):
        if maxiter is None:
            maxiter = 100
        
        print_vars = True
        plot_latent_space(self)
        
        self.optimize_by_tags(tags=set([VarTag.couplings]), maxiter=maxiter, print_vars=print_vars)
        self.optimize_by_tags(tags=set([VarTag.latent_x]), maxiter=maxiter, print_vars=print_vars)
        plot_latent_space(self)
        self.optimize_by_tags(tags=set([VarTag.couplings]), maxiter=maxiter, print_vars=print_vars)
        self.optimize_by_tags(tags=set([VarTag.kernel_params]), maxiter=maxiter, print_vars=print_vars)
        self.optimize_by_tags(tags=set([VarTag.couplings]), maxiter=maxiter, print_vars=print_vars)
        self.optimize_by_tags(tags=set([VarTag.augmenting_inputs]), maxiter=maxiter, print_vars=print_vars)
        plot_latent_space(self)
        self.optimize_by_tags(tags=set([VarTag.couplings]), maxiter=maxiter, print_vars=print_vars)


    def optimize_by_tags(self, tags=None, maxiter=None, print_vars=False):
        assert tags is not None
        if maxiter is None:
            maxiter = 100

        stored_non_differentiable = self.ns.non_differentiable
        self.ns.non_differentiable = set([var.symbol for var in self.ns.vars.values() if not tags & var.tags])
        varargs = [opt.MaxIterations(maxiter), opt.PrecisionFactor(1e3)]
        if self.param.estimation_mode == EstimationMode.ELBO:
            f_df = self.ns.make_function_and_gradient(self.neg_elbo, args=all)
            xOpt, f, d = opt.theano_optimize_bfgs_l(f_df, varargs=varargs)
            pl.info("Evaluated ELBO = {}".format(self.ns.evaluate(self.elbo)))
        elif self.param.estimation_mode == EstimationMode.MAP:
            f_df = self.ns.make_function_and_gradient(self.neg_loglikelihood, args=all)
            xOpt, f, d = opt.theano_optimize_bfgs_l(f_df, varargs=varargs)
            pl.info("Evaluated log-likelihood = {}".format(self.ns.evaluate(self.loglikelihood)))

        self.ns.non_differentiable = stored_non_differentiable
        if print_vars:
            self.print_by_tags(tags)
        return xOpt, f, d


    def save_state_to(self, filename):
        print("Writing state to {}".format(filename))
        with open(filename, "wb") as filehandle:
            cPickle.dump(self.ns.get_vars_state(), filehandle, protocol=cPickle.HIGHEST_PROTOCOL)


    def load_state_from(self, filename):        
        pl.info("Loading state from {}...".format(filename))
        vars_state = self.ns.get_vars_state()
        try:
            with open(filename, "rb") as filehandle:
                vars_state = cPickle.load(filehandle)
            self.ns.set_vars_state(vars_state)
            pl.info("State loading complete.")
        except IOError:
            self.ns.set_vars_state(vars_state)
            pl.info("State loading failed.")            
            raise


    def print_by_tags(self, tags=None):
        assert tags is not None
        for var in self.ns.vars.values():
            if tags & var.tags:
                pl.info(var.get_name())
                pl.info(self.ns.get_value(var))

    def precalc_posterior_predictive(self):
        if self.param.estimation_mode == EstimationMode.ELBO:
            self._precalc_posterior_predictive_elbo()
        elif self.param.estimation_mode == EstimationMode.MAP:
            self._precalc_posterior_predictive_map()
        W = self.param.nparts  # number of parts
        self.pp_dyn_gpfs = matrix_of_nones(W, W)
        for i in range(W):
            for j in range(W):
                self.pp_dyn_gpfs[j][i] = gpr.GP(self.dyn_kernel_objs[j][i], 
                                     Dx=self.param.parts[j].Q*self.param.dynamics_order, 
                                     Dy=self.param.parts[i].Q).sample_function()
                self.pp_dyn_gpfs[j][i].condition_on(self.pp_dyn_aug_z[j][i], self.pp_dyn_aug_u_means[j][i])
        self.pp_lvm_gpfs = list_of_nones(W)
        for i in range(W):
            self.pp_lvm_gpfs[i] = gpr.GP(self.lvm_kernel_objs[i],
                                          Dx=self.param.parts[i].Q,
                                          Dy=self.pp_lvm_aug_u_means[i].shape[1]).sample_function()
            self.pp_lvm_gpfs[i].condition_on(self.pp_lvm_aug_z[i], 
                                             self.pp_lvm_aug_u_means[i])
        
        

    def _precalc_posterior_predictive_elbo(self):
        W = self.param.nparts  # number of parts
        for i in range(W):
            FF_i = self.ns.evaluate(self.pp_dyn_FF_eq[i])  # [(WM)*(WM)]
            GG_i = self.ns.evaluate(self.pp_dyn_GG_eq[i])  # [(WM)*Q]
            Kzzdiag_i = self.ns.evaluate(self.pp_dyn_Kzzdiag_eq[i])  # [(WM)*(WM)]
            u_covars_i = np.linalg.inv((np.linalg.inv(Kzzdiag_i) + FF_i))  # [(WM)*(WM)]
            u_means_i = u_covars_i.dot(GG_i)  # [(WM)*Q]
            for j in range(W):
                M = self.param.parts[j].dyn_M  # number of augmenting points
                self.pp_dyn_aug_u_means[j][i] = u_means_i[j*M:j*M+M, :]
                self.pp_dyn_aug_u_covars[j][i] = u_covars_i[j*M:j*M+M, j*M:j*M+M]  # marginal covariance of u^{j, i}
                self.pp_dyn_Kauginv[j][i] = np.linalg.inv(Kzzdiag_i[j*M:j*M+M, j*M:j*M+M])
                self.pp_dyn_aug_z[j][i] = self.ns.get_value(self.dyn_aug_z[j][i])
        self.pp_alpha = self.ns.get_value(self.alpha)

        for i in range(W):
            self.pp_lvm_aug_z[i] = self.ns.get_value(self.lvm_aug_z[i])
            self.pp_lvm_Kauginv[i] = np.linalg.inv(self.ns.evaluate(self.lvm_Kaug[i]))
            self.pp_lvm_beta[i] = self.ns.get_value(self.lvm_beta[i])
            lvm_psi1 = self.ns.evaluate(self.lvm_psi1[i])
            lvm_psi2 = self.ns.evaluate(self.lvm_psi2[i])
            u_means_i, u_covars_i = vceq.optimal_q_u(self.pp_lvm_Kauginv[i], 
                                                     1.0 / self.pp_lvm_beta[i], 
                                                     self.param.parts[i].data.Y_centered_value, 
                                                     lvm_psi1,
                                                     lvm_psi2)
            self.pp_lvm_aug_u_means[i] = u_means_i
            self.pp_lvm_aug_u_covars[i] = u_covars_i
            
    def _precalc_posterior_predictive_map(self):
        W = self.param.nparts  # number of parts
        self.pp_alpha = self.ns.get_value(self.alpha)
        
        Xouts = list_of_nones(W)
        self.pp_dyn_Kxxs = matrix_of_nones(W, W)
        for i in range(W):
            self.pp_lvm_beta[i] = self.ns.get_value(self.lvm_beta[i])
            self.pp_lvm_aug_z[i] = self.ns.evaluate(self.x_means[i])
            self.pp_lvm_aug_u_means[i] = vceq.GPLVM.F_optimal(Kxx=self.ns.evaluate(self.lvm_Kxxs[i]),
                                                              y=self.param.parts[i].data.Y_centered_value,
                                                              beta=self.pp_lvm_beta[i])
            Xouts[i] = self.ns.evaluate(self.dyn_xt_means[i])
            for j in range(W):
                self.pp_dyn_Kxxs[j][i] = self.ns.evaluate(self.dyn_Kxxs[j][i])
                self.pp_dyn_aug_z[j][i] = self.ns.evaluate(self.dyn_xtminus_means[j])
        self.pp_dyn_aug_u_means = vceq.GPDM.F_optimal_coupled_kron(self.pp_dyn_Kxxs, 
                                                                   Xouts, 
                                                                   self.pp_alpha, 
                                                                   order=self.param.dynamics_order)
            
    def get_dynamics_start_point(self, timepoint=0):
        W = self.param.nparts  # number of parts
        x_path_start = list_of_nones(W)  # W*[nsteps*Q]

        # Make starting points for all parts
        for i in range(W):
            x_path_start[i] = self.ns.get_value(self.x_means[i])[timepoint:timepoint+2, :]  # 2-nd order dynamics
        return x_path_start


    def get_dynamics_start_point_by_training_chunk(self, training_chink_id=0, offset=0):
        timepoint = self.param.data.sequences_indexes[training_chink_id][offset]
        return self.get_dynamics_start_point(timepoint)


    def run_generative_dynamics(self, nsteps, startpoint=None):
        """
        Run GPDM in generative mode for nsteps.

        Args:
            nsteps (int): Number of steps to run.
            startpoint (None (W)[order*D] or (W)[L*order*D]):
        
        Returns:
            (W)[nsteps*D] or (W)[L*nsteps*D]: Generated trajectories.
        """
        if startpoint is None:
            startpoint = self.get_dynamics_start_point()
        
        W = self.param.nparts  # number of parts
        gpfs = matrix_of_nones(W, W)
        for i in range(W):
            for j in range(W):
                gpfs[j][i] = gpr.GP(self.dyn_kernel_objs[j][i], 
                                     Dx=self.param.parts[j].Q*self.param.dynamics_order, 
                                     Dy=self.param.parts[i].Q).sample_function()
                gpfs[j][i].condition_on(self.pp_dyn_aug_z[j][i], self.pp_dyn_aug_u_means[j][i])
        x_path = vceq.GPDM.generate_mean_prediction_coupled(gpfs=gpfs,
                                                            alpha=self.pp_alpha, 
                                                            x0s=startpoint, 
                                                            T=nsteps, 
                                                            order=self.param.dynamics_order)
        return x_path


    def lvm_map_to_observed(self, x):
        W = self.param.nparts  # number of parts
        y = list_of_nones(W)
        for i in range(W):
            y[i], _ = self.pp_lvm_gpfs[i].posterior_predictive(x[i])
            y[i] += self.param.parts[i].data.Y_mean_value
        return y    


def optimize_joint(model, maxiter=100, save_directory=None, prefix=None,
        iter_callback=default_model_plotter):

    statefilename = save_directory + "/iter({})-state.pkl".format(iter_callback.counter)
    try:
        model.load_state_from(statefilename)
    except IOError:
        model.optimize_by_tags(tags=set([VarTag.latent_x, 
                VarTag.augmenting_inputs, 
                VarTag.couplings, VarTag.kernel_params]), 
                maxiter=maxiter, print_vars=False)
        model.save_state_to(statefilename)

    model.precalc_posterior_predictive()
    iter_callback.save_dir = save_directory
    iter_callback(model)
    

def optimize_blocked(model, maxrun=10, maxiter=100, print_vars=False, 
        save_directory=None,
        optimize_augmenting_inputs_first=True,
        iter_callback=default_model_plotter):
        
    def get_f(model):
        if model.param.estimation_mode == EstimationMode.ELBO:
            return model.ns.evaluate(model.elbo)
        elif model.param.estimation_mode == EstimationMode.MAP:
            return model.ns.evaluate(model.loglikelihood)

    # Kernel parameters and couplings must be optimized together    
    
    f0 = get_f(model)
    for i in range(maxrun):
        statefilename = save_directory + "/iter({})-state.pkl".format(iter_callback.counter)
        try:
            model.load_state_from(statefilename)
            f = get_f(model)
        except IOError:
            if i == 0:
                # For some reason optimizing latent inputs first helps with matrix singularities
                tags = set([VarTag.couplings, VarTag.kernel_params])
                if optimize_augmenting_inputs_first:
                    tags.add(VarTag.augmenting_inputs)
                xOpt, f, d = model.optimize_by_tags(tags=tags, maxiter=maxiter, print_vars=print_vars)

            xOpt, f, d = model.optimize_by_tags(tags=set([VarTag.latent_x]), maxiter=maxiter, print_vars=False)
            xOpt, f, d = model.optimize_by_tags(tags=set([VarTag.couplings, VarTag.kernel_params]), maxiter=maxiter, print_vars=print_vars)
            if model.param.estimation_mode == EstimationMode.ELBO:
                xOpt, f, d = model.optimize_by_tags(tags=set([VarTag.augmenting_inputs]), maxiter=maxiter, print_vars=False)
                xOpt, f, d = model.optimize_by_tags(tags=set([VarTag.couplings, VarTag.kernel_params]), maxiter=maxiter, print_vars=print_vars)
        
            if save_directory is not None:
                model.save_state_to(statefilename)                
   
        model.precalc_posterior_predictive()
        iter_callback.save_dir = save_directory
        iter_callback(model)
        
        if np.allclose(f, f0):
            return
        
        f0 = f

            

