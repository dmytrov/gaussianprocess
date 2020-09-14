import os
import time
import collections
from six.moves import cPickle
import numpy as np
import numerical.numpytheano as nt
import numerical.numpyext.linalg as ntla
import numerical.numpyext as npx
import numerical.numpyext.dimreduction as dr
import theano
import ml.gptheano.vecgpdm.equations as vceq
import numerical.numpytheano.theanopool as tp
import ml.gptheano.vecgpdm.kernels as krn
import ml.gptheano.vecgpdm.gaussianprocess as gpr
import ml.gp.equations as gpeq
import numerical.theanoext as thx
import matplotlibex.mlplot as plx
import matplotlib.pyplot as plt
import ml.gptheano.vecgpdm.optimization as opt
from ml.gptheano.vecgpdm.equations import list_of_nones
from ml.gptheano.vecgpdm.equations import matrix_of_nones
from ml.gptheano.vecgpdm.enums import *

"""
Model of an explicitly coupled dynamics with separate exchangeable primitives
and learnable interacions
"""




class ELBOMode(object):
    # Include cross-couplings too. 
    # Almost equivalent to vCGPDM, 
    # but sross-couplings are shared between different MPs
    full = 0

    # Only separate MP dynamics. 
    # Latent dynamics is independent. 
    # One LVM per body part
    separate_dynamics = 1   
    
    # Construct ELBO with cross-couplings from the diagonal matrix 
    # learned with separate_dynamics mode,
    # optimize couplings only
    couplings_only = 2   


def remove_nones(struct):
    """
    Compacts the struc (a list of lists) by removing the NaNs.
    Non-existing interactions are marked as NaNs and have to be removed.
    """
    if isinstance(struct, list):
        res = [remove_nones(item) for item in struct if remove_nones(item) is not None]
        if len(res) > 0:
            return res
        else:
            return None
    else:
        return struct

def remove_trend(y):
    y_bias = np.mean(y, axis=0)
    x = np.linspace(-1, 1, num= y.shape[0])
    y_centered = y - y_bias
    a = y.T.dot(x) / x.dot(x)
    res = y - a*x[:, np.newaxis]
    return res

class ObservedMeanMode(object):
    per_part = 0
    per_segment = 1
    #per_MP_type = 2  
    #per_piece = 3  



class MPToPartInfluence(object):  
    """
    motor primitive -> part influnce
    Contains augmenting inputs and kernel matrix for dynamics
    """
    def __init__(self, mp_type, part_type, dyn_kern, ns=nt.NumpyLinalg):
        self.ns = ns
        self.mp_type = mp_type  # back reference
        self.part_type = part_type
        self.M = None
        self.aug_z = None  # augmenting inputs 
        self.aug_z_centered = None  # centered augmenting inputs 
        self.dyn_kern = dyn_kern  # kernel object
        self.Kaug = None  # kernel matrix
        self.pp_aug_z_centered = None
        
        # log-likelihood related variables
        self.map_indexes = None  # indexes of mappings to access the X'es
        self.xin = None  # [T, D*order] - GP mapping inputs
        self.xout = None  # [T, D] - GP mapping outputs
        self.ll_K = None  # Gram matrix for map_indexes
        #self.ll_alphafull  # expanded noise variances to match the part_type segments
        #self.ll_Kfull = None  # expanded kernel to match the part_type segments
        #self.ll_pp_xin = None
        #self.ll_pp_Kinv = None
        #self.ll_pp_Fopt = None  # F_optimal

       
    @staticmethod
    def make_key(mp_type, part_type):
        return mp_type.part_type.name + "." + mp_type.name + "-to-" + part_type.name

    def get_key(self):
        return MPToPartInfluence.make_key(self.mp_type, self.part_type)

    def init_aug_z(self, dataset, M):
        self.M = M
        ds_ind = self.mp_type.get_ds_xtminus_indexes(dataset)
        order = len(ds_ind)
        nmappings = ds_ind[0].shape[0]
        
        aug_indexes = np.concatenate([[0], np.random.choice(a=range(1, nmappings), size=self.M-1, replace=False)])
        #aug_indexes = np.arange(0, nmappings, int(nmappings/M))[:M]
        
        aug_z = np.hstack([self.ns.get_value(dataset.x_means[self.mp_type.part_type.index])[ds_ind[i][aug_indexes], :] for i in range(order)])
        self.aug_z = self.ns.matrix("aug_z[{}]".format(self.get_key()), aug_z, bounds=(-10.0, 10.0), tags=(VarTag.augmenting_inputs))
        self.aug_z_centered = self.aug_z - self.ns.concatenate([self.mp_type.data_mean for i in range(order)])
        self.Kaug = self.dyn_kern.gram_matrix(self.aug_z_centered, self.aug_z_centered, self.ns)
        
    def precalc_posterior_predictive(self):
        self.pp_aug_z_centered = self.ns.evaluate(self.aug_z_centered)

    def ll_set_data(self, dataset):
        """
        Log-likelihood related.
        """
        self.map_indexes = self.mp_type.get_ds_ind(dataset)
        x = dataset.x_means[self.mp_type.part_type.index]
        self.xin = self.ns.concatenate([x[dataset.x_in_indexes[self.map_indexes, i]] for i in range(dataset.x_in_indexes.shape[1])], axis=1)
        #print "self.xin", self.xin 
        self.xout = x[dataset.x_out_indexes[self.map_indexes]]
        self.ll_K = self.dyn_kern.gram_matrix(self.xin, self.xin, ns=dataset.ns)

class MPToMPInfluence(object):  
    """ 
    motor primitive -> motor primitive influnce
    Contains only learned coupling (alpha).
    """
    def __init__(self, mp_type1, mp_type2, ns=nt.NumpyLinalg):
        self.ns = ns
        self.mp_type1 = mp_type1
        self.mp_type2 = mp_type2
        key = self.get_key()
        if mp_type1 is mp_type2:
            alpha_init = 1.0
        else:
            alpha_init = 2.0
        self.alpha = ns.scalar("alpha[{}]".format(key), alpha_init, bounds=(1.0e-2, 10.0), tags=(VarTag.couplings))  # prediction (influence) noise covariance
        self.optimize_post_training = False  # for those MPs influences on which are to be inferred indirectly, after the training, optimizing the ELBO

    @staticmethod
    def make_key(mp_type1, mp_type2):
        return mp_type1.part_type.name + "." + mp_type1.name + "-to-" + mp_type2.part_type.name + "." + mp_type2.name

    def get_key(self):
        return MPToMPInfluence.make_key(self.mp_type1, self.mp_type2)


class PartType(object):
    """
    Body part type, like upper, lower, hand, head, leg etc.
    Contains GPLVM ELBO for this part.
    Can generate observed data, X->Y mapping.
    """
    def __init__(self, name, dataset, index, y_indexes, Q, M, ns=nt.NumpyLinalg):
        self.ns = ns
        self.name = name
        self.dataset = dataset  # back reference
        self.index = index  # order in the list of part types
        self.y_indexes = y_indexes
        self.Q = Q  # latent X dimensionality
        self.M = M  # number of augmenting points
        self.mp_types = []  # MP types defined for this part

        # ELBO related variables
        self.lvm_log_beta = None
        self.lvm_beta = None
        self.lvm_kernel_obj = None  # kernel objects
        self.lvm_aug_z = None  # list of augmenting (inducing) inputs for all W parts
        self.lvm_Kaug = None  # kernel matrices of augmenting inputs
        self.lvm_psi0 = None 
        self.lvm_psi1 = None
        self.lvm_psi2 = None
        self.elbo = None  # LVM ELBO
        
        # Posterior-predictive variables
        self.pp_lvm_aug_z = None
        self.pp_lvm_Kauginv = None
        self.pp_lvm_beta = None
        self.pp_lvm_aug_u_means = None
        self.pp_lvm_aug_u_covars = None

        # LVM log-likelihood related variables
        self.loglikelihood = None
        self.ll_lvm_Kxx = None
        self.all_map_indexes = None  # [N] 
        #self.ll_Ks = None  # 
        #self.ll_C = None  #
        
    def init_x(self):
        i = self.index
        ds_ind = self.ns.int_vector("part_ds_ind[{}]".format(i), self.get_ds_ind())
        self.x_means = self.dataset.x_means[i][ds_ind, :]
        self.x_covars = self.dataset.x_covars_diags[i][ds_ind, :]
        self.y_centered = self.dataset.y_centered[i][ds_ind, :]
        
    def init_aug_z(self):
        print("PartType.init_aug_z")
        suffix = "_[{}]".format(self.name)
        nmappings = len(self.get_ds_ind())

        aug_indexes = np.concatenate([[0], np.random.choice(a=range(1, nmappings), size=self.M-1, replace=False)])
        #aug_indexes = np.arange(0, nmappings, int(nmappings/self.M))[:self.M]

        lvm_aug_z = self.ns.evaluate(self.x_means)[aug_indexes, :]
        self.lvm_beta = self.ns.scalar("lvm_beta"+suffix, 1.0, bounds=(1.0e-2, 1.0e4), tags=(VarTag.kernel_params))  # Precision !!! INVERSE !!! of the x->y mapping noise
        kern_width = 1.0
        self.lvm_aug_z = self.ns.matrix("lvm_aug_z"+suffix, lvm_aug_z, bounds=(-10.0, 10.0), tags=(VarTag.augmenting_inputs))
        self.lvm_kernel_obj = self.dataset.lvm_kern_type(self.Q, kern_width, suffix, ns=self.ns)
        self.lvm_Kaug = self.lvm_kernel_obj.gram_matrix(self.lvm_aug_z, self.lvm_aug_z, ns=self.ns)

        #x_means = self.ns.evaluate(self.x_means)
        #print(x_means)
        #print(lvm_aug_z)
        #print(self.name)
        #plt.figure()
        #plt.plot(x_means[:, 0], x_means[:, 1], '-', alpha=0.2)
        #plt.plot(lvm_aug_z[:, 0], lvm_aug_z[:, 1], '+', markersize=15)
        #plt.title(self.name)
        #plt.show()
        
    def init_psi_stats(self):
        self.lvm_psi0 = self.lvm_kernel_obj.psi_stat_0(self.x_means, self.x_covars)
        self.lvm_psi1 = self.lvm_kernel_obj.psi_stat_1(self.lvm_aug_z, self.x_means, self.x_covars)
        self.lvm_psi2 = self.lvm_kernel_obj.psi_stat_2(self.lvm_aug_z, self.x_means, self.x_covars)
        
    def init_elbo(self):
        self.elbo = vceq.GPLVM.variational_elbo(self.y_centered, 
                                               self.lvm_beta,
                                               self.lvm_Kaug, 
                                               self.lvm_psi0,
                                               self.lvm_psi1,
                                               self.lvm_psi2,
                                               ns=self.ns)
        #self.loglikelihood = vceq.GPLVM.loglikelihood(self.y_centered,
        #                                              self.lvm_beta, 
        #                                              self.lvm_kernel_obj.gram_matrix(self.x_means, self.x_means, ns=self.ns))
        

    def precalc_posterior_predictive(self):  # only LVM posterior-predictive
        self.pp_lvm_aug_z = self.ns.get_value(self.lvm_aug_z)
        self.pp_lvm_Kauginv = np.linalg.inv(self.ns.evaluate(self.lvm_Kaug))
        self.pp_lvm_beta = self.ns.get_value(self.lvm_beta)
        self.pp_lvm_aug_u_means, self.pp_lvm_aug_u_covars = vceq.optimal_q_u(self.pp_lvm_Kauginv, 
                                                                            1.0 / self.pp_lvm_beta, 
                                                                            self.ns.evaluate(self.y_centered), 
                                                                            self.ns.evaluate(self.lvm_psi1),
                                                                            self.ns.evaluate(self.lvm_psi2))

    def get_ds_ind(self):  # segmented data points indexes
        inds = []
        for trial in self.dataset.trials:
            for segment in trial.mp_segments:
                if segment.mp_type.part_type == self:
                    inds.append(segment.get_ds_ind())
        return np.hstack(inds)

    def init_loglikelihood(self):
        x_means = self.dataset.x_means[self.index][self.get_ds_ind(), :]
        y_centered = self.dataset.y_centered[self.index][self.get_ds_ind(), :]
        self.lvm_Kxx = self.lvm_kernel_obj.gram_matrix(x_means, x_means, ns=self.ns)
        self.loglikelihood = vceq.GPLVM.loglikelihood(Kxx=self.lvm_Kxx,
                                                      y=y_centered,
                                                      beta=self.lvm_beta,
                                                      ns=self.ns)

    def ll_precalc_posterior_predictive(self):
        x_means = self.ns.evaluate(self.dataset.x_means[self.index][self.get_ds_ind(), :])
        y_centered = self.ns.evaluate(self.dataset.y_centered[self.index][self.get_ds_ind(), :])
        self.pp_lvm_beta = self.ns.get_value(self.lvm_beta)
        pp_ll_f_optimal = vceq.GPLVM.F_optimal(Kxx=self.ns.evaluate(self.lvm_Kxx), y=y_centered, beta=self.pp_lvm_beta)
        self.pp_lvm_gpf = gpr.GP(self.lvm_kernel_obj,
                                  Dx=self.Q,
                                  Dy=y_centered.shape[1]).sample_function()
        self.pp_lvm_gpf.condition_on(x_means, pp_ll_f_optimal)

    def ll_map_to_observed(self, x):
        ystarmean, _ = self.pp_lvm_gpf.posterior_predictive(x)
        return ystarmean + self.dataset.y_means[self.index]

        
class MPType(object):
    """
    Motion primitive type associated with some part type.
    Has data mean value.
    """
    def __init__(self, name, part_type, remove_trend=False, ns=nt.NumpyLinalg):
        super(MPType, self).__init__()
        self.ns = ns
        self.name = name
        assert isinstance(part_type, PartType)
        self.part_type = part_type
        self.data_mean = self.ns.vector("data_mean[{}.{}]".format(self.part_type.name, self.name), 
                                        value=np.zeros([self.part_type.Q]), 
                                        differentiable=False,
                                        tags=VarTag.data_mean)
        self.pp_data_mean = None
        self.remove_trend = remove_trend

    def get_segments(self, dataset):
        segments = []
        for trial in dataset.trials:
            for segment in trial.mp_segments:
                if segment.mp_type is self:
                    segments.append(segment)
        return segments

    def get_pieces(self, dataset):
        pieces = []
        for piece in dataset.pieces:
            if self in piece.mps:
                pieces.append(piece)
        return pieces

    def get_ds_ind(self, dataset):
        return np.hstack([segment.get_ds_ind() for segment in self.get_segments(dataset)])

    def get_ds_xtminus_indexes(self, dataset):
        return [np.hstack([dataset.ns.get_value(piece.ds_xtminus_indexes[i]) for piece in self.get_pieces(dataset)]) for i in range(2)]
        
    def update_data_mean(self, dataset):
        if len(self.get_segments(dataset)) > 0:
            self.ns.set_value(self.data_mean, np.mean(self.get_x_means(dataset), axis=0))
            #print("Updating mean: {} = {}".format("data_mean[{}.{}]".format(self.part_type.name, self.name), 
            #                              self.ns.get_value(self.data_mean)))

    def get_x_means(self, dataset):
        if len(self.get_segments(dataset)) > 0:
            mp_data = dataset.x_means[self.part_type.index]
            return self.ns.get_value(mp_data)[self.get_ds_ind(dataset), :]
        return None

    def get_y(self, dataset):
        if len(self.get_segments(dataset)) > 0:
            return dataset.y[self.part_type.index][self.get_ds_ind(dataset), :]
        return None
    
    def precalc_posterior_predictive(self):
        self.pp_data_mean = self.ns.get_value(self.data_mean)

class MPSegment(object):  
    """
    Marked segment with a MP.
    Used to construct the dataset, marks a piece of a trial data as a MP.
    """
    def __init__(self, mp_type, start, end, trial=None, ns=nt.NumpyLinalg):
        super(MPSegment, self).__init__()
        self.ns = ns
        self.mp_type = mp_type
        self.tr_start = start  # start frame, absolute index in Y data chunk of the trial
        self.tr_end = end  # end frame, absolute index index in Y data chunk of the trial
        self.trial = trial  # back-refecence to the trial
        self.y_mean = None

    def get_tr_ind(self):  # indexes of the data in the trial
        return np.arange(self.tr_start, self.tr_end)

    def get_ds_ind(self):  # indexes of the data in the full dataset
        return self.get_tr_ind() + self.trial.offset()



def diag_to_matrix(a):
    """
    List of lists to diagonal to full matrix.
    :param a: [N][N] list of lists
    :return: [N][N]. res[i][j] = a[i][i]
    """
    N = len(a)
    res = matrix_of_nones(N, N)
    for i in range(N):
        for j in range(N):
            res[i][j] = a[i][i]
    return res   



class Piece(object):  
    """
    Continuous piece of interacting MPs. 
    Used to construct the ELBO.
    Many pieces form the full ELBO.
    """
    def __init__(self, mps, starts_ends, trial, index, dynamics_order=2, ns=nt.NumpyLinalg):
        print("Creating a piece. starts_ends={}".format(starts_ends))
        #assert start == 0 or start > dynamics_order  # cases in between make not much sense
        self.ns = ns
        self.mps = mps  # list of W MPs, one MP per part
        if not isinstance(starts_ends[0], list):
            starts_ends = list(starts_ends)
        self.nchunks = len(starts_ends)
        self.starts_ends = starts_ends
        self.tr_starts = [se[0] for se in starts_ends]  # start frame, absolute index in Y data chunk of the trial
        self.tr_ends = [se[1] for se in starts_ends]  # end frame, absolute index index in Y data chunk of the trial
        self.trial = trial  # back-refecence to the trial to access influences
        self.index = index
        self.dynamics_order = dynamics_order
        self.suffix = "({}.{})".format(self.trial.index, self.index)
        
        #self.tr_xt_indexes = np.arange(self.tr_start, self.tr_end)  # continuous array of indexes to access the trial data cunk. The piece is defined on this range
        self.has_x0 = True
        #if self.has_x0:  # no previous state for the dynamics
        #    self.tr_start = dynamics_order  # indentation from zero start

        self.ds_x0_indexes = [se[0] for se in self.starts_ends]
        self.ds_xtminus_indexes = [ns.int_vector("ds_xtminus_indexes({})({})".format(self.suffix, i), 
                                                 np.hstack([np.arange(se[0], se[1] - dynamics_order) for se in self.starts_ends])  + i + self.offset()) 
                                   for i in range(dynamics_order)]  # from (start - dyn_order) to (end - dyn_order) of the piece
        self.ds_xtplus_indexes = ns.int_vector("ds_xtplus_indexes({})".format(self.suffix), 
                                               np.hstack([np.arange(se[0] + dynamics_order, se[1]) for se in self.starts_ends]) + self.offset()) # from start to end of the piece
        self.x_means_centered = None # [W][QN,D]
        self.x_covars = None # [W][QN,D]
        self.xtminus_centered = None  # [W][QN,D]
        self.xtminus_covars = None  # [W][QN,D]
        self.xtplus_centered = None  # [W][N,D]
        self.xtplus_covars = None  # [W][N,D]
        self.mp_to_part_influences = None  # [W][W]
        self.Kaug = None  # [W,W][...]  kernel matrix
        self.psi0 = None  # [W,W][...]
        self.psi1 = None  # [W,W][...]
        self.psi2 = None  # [W,W][...]
        self.alpha = None  # [W,W]
        self.elbo = None
        self.elbo_mode = None
        self.loglikelihood = None
        self.pp_mp_to_part_influences = None
        self.pp_dyn_FF_eq = None
        self.pp_dyn_GG_eq = None
        self.pp_dyn_Kzzdiag_eq = None
        self.pp_dyn_aug_u_means_centered = None
        self.pp_dyn_Kauginv = None
        self.pp_dyn_aug_z = None
        self.pp_alpha = None
        
    def offset(self):  # offset in the combined dataset
        return self.trial.offset()

    def init_x(self):
        W = len(self.trial.dataset.part_types)
        self.x_means_centered = list_of_nones(W)
        self.x_covars = list_of_nones(W)
        self.xtminus_centered = list_of_nones(W)
        self.xtplus_centered = list_of_nones(W)
        self.xtminus_covars = list_of_nones(W)
        self.xtplus_covars = list_of_nones(W)
        for mp in self.mps:
            i = mp.part_type.index
            self.x_means_centered[i] = self.trial.dataset.x_means[i] - mp.data_mean
            self.x_covars[i] = self.trial.dataset.x_covars_diags[i]
            self.xtminus_centered[i] = self.ns.concatenate([self.x_means_centered[i][self.ds_xtminus_indexes[j], :] for j in range(self.dynamics_order)], axis=1)
            self.xtminus_covars[i] = self.ns.concatenate([self.trial.dataset.x_covars_diags[i][self.ds_xtminus_indexes[j], :] for j in range(self.dynamics_order)], axis=1)
            self.xtplus_centered[i] = self.x_means_centered[i][self.ds_xtplus_indexes, :]
            self.xtplus_covars[i] = self.trial.dataset.x_covars_diags[i][self.ds_xtplus_indexes, :]
        
    def init_psi_stats(self):
        W = len(self.trial.dataset.part_types)
        self.Kaug = matrix_of_nones(W, W)
        self.psi0 = matrix_of_nones(W, W)
        self.psi1 = matrix_of_nones(W, W)
        self.psi2 = matrix_of_nones(W, W)
        self.alpha = matrix_of_nones(W, W)
        self.mp_to_part_influences = matrix_of_nones(W, W)
        for mp_i in self.mps:
            for mp_j in self.mps:
                i = mp_i.part_type.index
                j = mp_j.part_type.index
                self.mp_to_part_influences[j][i] = self.trial.dataset.part_influence[MPToPartInfluence.make_key(mp_j, mp_i.part_type)]
                self.Kaug[j][i] = self.mp_to_part_influences[j][i].Kaug
                self.psi0[j][i] = self.mp_to_part_influences[j][i].dyn_kern.psi_stat_0(self.xtminus_centered[j], self.xtminus_covars[j])
                self.psi1[j][i] = self.mp_to_part_influences[j][i].dyn_kern.psi_stat_1(self.mp_to_part_influences[j][i].aug_z_centered, self.xtminus_centered[j], self.xtminus_covars[j])
                self.psi2[j][i] = self.mp_to_part_influences[j][i].dyn_kern.psi_stat_2(self.mp_to_part_influences[j][i].aug_z_centered, self.xtminus_centered[j], self.xtminus_covars[j])
                ji_mp_influence = self.trial.dataset.mp_influence[MPToMPInfluence.make_key(mp_j, mp_i)]
                self.alpha[j][i] = ji_mp_influence.alpha
                
    def init_elbo(self, elbo_mode=ELBOMode.full):
        self.elbo_mode = elbo_mode
        if self.elbo_mode == ELBOMode.full:
            self.init_elbo_full()
        elif self.elbo_mode == ELBOMode.separate_dynamics:
            self.init_elbo_separate_dynamics()
        elif self.elbo_mode == ELBOMode.couplings_only:
            self.init_elbo_couplings_ony()

    def init_elbo_full(self):  # parts are coupled with {part,MP}->{part} influences
        self.pp_mp_to_part_influences = self.mp_to_part_influences
        self.elbo, (self.pp_dyn_FF_eq, self.pp_dyn_GG_eq, self.pp_dyn_Kzzdiag_eq) = vceq.GPDM.variational_explicit_elbo_diag_x_covars(
            remove_nones(self.xtplus_centered), 
            remove_nones(self.xtplus_covars), 
            self.ns.stacklists(remove_nones(self.alpha)),
            remove_nones(self.Kaug),
            remove_nones(self.psi0),
            remove_nones(self.psi1),
            remove_nones(self.psi2),
            ns=self.ns)
        self.add_elbo_x0_cross_entropy()

    def init_elbo_separate_dynamics(self):  # no parts are interacting, completely factorized dynamics
        W = len(self.trial.dataset.part_types)
        self.elbo = 0.0
        self.pp_dyn_FF_eq = list_of_nones(W)
        self.pp_dyn_GG_eq = list_of_nones(W)
        self.pp_dyn_Kzzdiag_eq = list_of_nones(W)
        self.pp_mp_to_part_influences = self.mp_to_part_influences
        for mp in self.mps:
            i = mp.part_type.index
            if self.Kaug[i][i] is not None:
                elbo, (pp_dyn_FF_eq, pp_dyn_GG_eq, pp_dyn_Kzzdiag_eq) = vceq.GPDM.variational_explicit_elbo_diag_x_covars(
                    [self.xtplus_centered[i]],
                    [self.xtplus_covars[i]], 
                    self.ns.stacklists([[self.alpha[i][i]]]),
                    [[self.Kaug[i][i]]],
                    [[self.psi0[i][i]]],
                    [[self.psi1[i][i]]],
                    [[self.psi2[i][i]]],
                    ns=self.ns)
                self.elbo += elbo
                self.pp_dyn_FF_eq[i] = pp_dyn_FF_eq[0]
                self.pp_dyn_GG_eq[i] = pp_dyn_GG_eq[0]
                self.pp_dyn_Kzzdiag_eq[i] = pp_dyn_Kzzdiag_eq[0]
        self.add_elbo_x0_cross_entropy()
        
        self.loglikelihood = 0.0

    def init_elbo_couplings_ony(self):
        # Re-initialize the couplings
        for mp_i in self.mps:
            for mp_j in self.mps:
                i = mp_i.part_type.index
                j = mp_j.part_type.index
                ji_mp_influence = self.trial.dataset.mp_influence[MPToMPInfluence.make_key(mp_j, mp_i)]
                self.ns.set_value(ji_mp_influence.alpha, 1.0)

        # There are no cross-GP-mapping inducing points, only i->i mappings, reused for j->i mappings
        Kaug = diag_to_matrix(remove_nones(self.Kaug))
        psi0 = diag_to_matrix(remove_nones(self.psi0))
        psi1 = diag_to_matrix(remove_nones(self.psi1))
        psi2 = diag_to_matrix(remove_nones(self.psi2))
        self.pp_mp_to_part_influences = diag_to_matrix(remove_nones(self.mp_to_part_influences))
        self.elbo, (self.pp_dyn_FF_eq, self.pp_dyn_GG_eq, self.pp_dyn_Kzzdiag_eq) = vceq.GPDM.variational_explicit_elbo_diag_x_covars(
            remove_nones(self.xtplus_centered), 
            remove_nones(self.xtplus_covars), 
            self.ns.stacklists(remove_nones(self.alpha)),
            Kaug,
            psi0,
            psi1,
            psi2,
            mode=self.elbo_mode,
            ns=self.ns)
        self.add_elbo_x0_cross_entropy()

    def add_elbo_x0_cross_entropy(self):
        s = np.cumsum([se[1] - se[0] for se in self.starts_ends])
        x0_inds = np.hstack([[0], s[:-1]])
        if self.has_x0:
            for mp in self.mps:
                i = mp.part_type.index
                #print(self.ns.evaluate(self.xtminus_centered[i]).shape)
                for x0_ind in x0_inds:
                    self.elbo += vceq.gaussians_cross_entropy(self.x_means_centered[i][x0_ind, :], 
                                                              self.ns.diag(self.x_covars[i][x0_ind, :]), 
                                                              self.ns.zeros_like(self.x_means_centered[i][x0_ind, :]), 
                                                              self.ns.diag(self.ns.ones_like(self.x_covars[i][x0_ind, :])),
                                                              ns=self.ns)
                    self.elbo += vceq.gaussians_cross_entropy(self.x_means_centered[i][x0_ind + 1, :], 
                                                              self.ns.diag(self.x_covars[i][x0_ind + 1, :]), 
                                                              self.x_means_centered[i][x0_ind, :], 
                                                              self.ns.diag(self.ns.ones_like(self.x_covars[i][x0_ind, :])),
                                                              ns=self.ns)

    def precalc_posterior_predictive(self):
        if self.elbo_mode == ELBOMode.full:
            self.precalc_posterior_predictive_full()
        elif self.elbo_mode == ELBOMode.separate_dynamics:
            self.precalc_posterior_predictive_separate_dynamics()
        elif self.elbo_mode == ELBOMode.couplings_only:
            self.precalc_posterior_predictive_couplings_ony()

    def precalc_posterior_predictive_full(self):
        W = len(self.trial.dataset.part_types)
        self.pp_dyn_aug_u_means_centered = matrix_of_nones(W, W)
        self.pp_dyn_Kauginv = matrix_of_nones(W, W)
        self.pp_dyn_aug_z_centered = matrix_of_nones(W, W)
        for mp_i in self.mps:
            i = mp_i.part_type.index
            FF_i = self.ns.evaluate(self.pp_dyn_FF_eq[i])  # [(WM)*(WM)]
            GG_i = self.ns.evaluate(self.pp_dyn_GG_eq[i])  # [(WM)*Q]
            Kzzdiag_i = self.ns.evaluate(self.pp_dyn_Kzzdiag_eq[i])  # [(WM)*(WM)]
            u_covars_i = np.linalg.inv((np.linalg.inv(Kzzdiag_i) + FF_i))  # [(WM)*(WM)]
            u_means_i = u_covars_i.dot(GG_i)  # [(WM)*Q]
            for mp_j in self.mps:
                j = mp_j.part_type.index
                M = self.pp_mp_to_part_influences[j][i].M
                self.pp_dyn_aug_u_means_centered[j][i] = u_means_i[j*M:j*M+M, :]
                self.pp_dyn_Kauginv[j][i] = np.linalg.inv(Kzzdiag_i[j*M:j*M+M, j*M:j*M+M])
                self.pp_dyn_aug_z_centered[j][i] = self.ns.evaluate(self.pp_mp_to_part_influences[j][i].aug_z_centered)
                #if self.elbo_mode == vceq.ELBOMode.full:
                #    self.pp_dyn_aug_z_centered[j][i] = self.ns.evaluate(self.pp_mp_to_part_influences[j][i].aug_z_centered)
                #else:
                #    self.pp_dyn_aug_z_centered[j][i] = self.ns.evaluate(self.mp_to_part_influences[j][j].aug_z_centered)
        self.pp_alpha = self.ns.evaluate(self.ns.stacklists(self.alpha))
        

    def precalc_posterior_predictive_separate_dynamics(self):
        self.precalc_posterior_predictive_full()

    def precalc_posterior_predictive_couplings_ony(self):
        self.precalc_posterior_predictive_full()
        

    def save_plot_piece_posterior_predictive(self, directory):
        for mp_i in self.mps:
            i = mp_i.part_type.index
            for mp_j in self.mps:
                j = mp_j.part_type.index
                fig = plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                x = mp_j.get_x_means(self.trial.dataset)
                plt.plot(x[:, 0], x[:, 1], '-', alpha=0.1)
                x = self.ns.evaluate(self.xtplus_centered[j]) + mp_j.pp_data_mean
                z = self.pp_dyn_aug_z_centered[j][i] + np.hstack([mp_j.pp_data_mean, mp_j.pp_data_mean])
                plt.plot(x[:, 0], x[:, 1], '-', alpha=0.3)
                plt.plot(z[:, 0], z[:, 1], 'o', markersize=15, markeredgewidth=2, fillstyle="none")

                plt.subplot(1, 2, 2)
                x = mp_i.get_x_means(self.trial.dataset)
                plt.plot(x[:, 0], x[:, 1], '-', alpha=0.1)
                x = self.ns.evaluate(self.xtplus_centered[i]) + mp_i.pp_data_mean
                u = self.pp_dyn_aug_u_means_centered[j][i] + mp_i.pp_data_mean
                plt.plot(x[:, 0], x[:, 1], '-', alpha=0.3)
                plt.plot(u[:, 0], u[:, 1], 'o', markersize=15, markeredgewidth=2, fillstyle="none")

                key = MPToPartInfluence.make_key(mp_j, mp_i.part_type)
                plt.savefig("{}/pp_{}.pdf".format(directory, key))
                plt.close(fig)
        pass        

    def get_segments(self):  # segments defining the first chunk
        res = []
        start, end = self.starts_ends[0]
        for segment in self.trial.mp_segments:
            if segment.tr_start <= start and segment.tr_end >= end:
                res.append(segment)
        return res


class Trial(object):  # converts segmentation description of a trial into pieces
    def __init__(self, dataset, Y, index, ns=nt.NumpyLinalg):
        self.ns = ns
        self.Y = Y.copy()  # trial data
        self.npoints = self.Y.shape[0]
        self.dataset = dataset  # back-refecence to the complete dataset to access influences
        self.mp_segments = []  # list of MPSegment, provided
        self.pieces = []  # list of Piece. Used later to construct the ELBO
        self.index = index  # order in the list

    def add_segment(self, segment):
        self.mp_segments.append(segment)
        segment.trial = self

    def remove_irrelevant_data(self):
        npoints = self.Y.shape[0]
        marked = np.zeros([npoints], dtype=bool)
        for segment in self.mp_segments:
            marked[segment.tr_start:segment.tr_end] = True
        irr_se = []
        f = 1  # 1 - waiting for irrelevant; 2 - waiting for relevant data
        start = 0
        for i, m in enumerate(marked):
            if f == 1 and not m:
                start = i
                f = 2
            elif f == 2 and m:
                end = i
                irr_se.append([start, end])
                f = 1
        if f == 2 and npoints > start:
            irr_se.append([start, npoints])
        irr_size = [se[1] - se[0] for se in irr_se]
        irr_cumsum = np.cumsum(irr_size)
        for segment in self.mp_segments:
            i = None
            for j, se in enumerate(irr_se):
                if segment.tr_start >= se[1]:
                    i = j
            if i is not None:
                segment.tr_start -= irr_cumsum[i]
                segment.tr_end -= irr_cumsum[i]
        self.Y = self.Y[marked, :]
        self.npoints = self.Y.shape[0]

    def remove_trends_in_segments(self):
        for segment in self.mp_segments:
            if segment.mp_type.remove_trend:
                self.Y[segment.tr_start:segment.tr_end, segment.mp_type.part_type.y_indexes] = \
                    remove_trend(self.Y[segment.tr_start:segment.tr_end, segment.mp_type.part_type.y_indexes])

    def create_pieces(self):
        changepoints = []
        changepoints.extend([segment.tr_start for segment in self.mp_segments])
        changepoints.extend([segment.tr_end for segment in self.mp_segments])
        changepoints = list(set(changepoints))
        changepoints.sort()
        pieces_mps = []
        pieces_starts_ends = []
        #pieces_indexes = []
        for i in range(len(changepoints)-1):
            start = changepoints[i]
            end = changepoints[i+1]
            # Find all MPSegments overlapping with this piece
            mps = []
            for segment in self.mp_segments:
                if segment.tr_start <= start and segment.tr_end >= end:
                    mps.append(segment.mp_type)
            if len(mps) > 0:
                pieces_mps.append(mps)
                pieces_starts_ends.append([start, end])
                #piece = Piece(mps, [start, end], self, index=i, ns=self.ns)
                #self.pieces.append(piece)
        while len(pieces_mps) > 0:
            print("Trial {}".format(self.index))
            mps = pieces_mps[0]
            ii = [i for i, m in enumerate(pieces_mps) if set(m) == set(mps)]
            notii = [i for i, m in enumerate(pieces_mps) if set(m) != set(mps)]
            piece = Piece(mps, [pieces_starts_ends[i] for i in ii], self, index=len(self.pieces), ns=self.ns)
            self.pieces.append(piece)
            pieces_mps = [pieces_mps[i] for i in notii]
        return self.pieces
        
    def offset(self):  # offset in the combined dataset
        return self.dataset.trials_offsets[self.index]

def intersect(a, b):
    return np.intersect1d(a, b)

def index_of_in(a, b):
    """
    a = [1, 2, 3, 4]
    b = [2, 4, 5, 6]
    index_of_in(a, b) == [1, 3, -1, -1]
    """
    al = list(a)
    res = [al.index(bitem) if bitem in al else -1 for bitem in b]
    return res

def expand_kernel(ksize, Kfrom, noisevars, indexes_from_to, indexes_to, ns=nt.NumpyLinalg):
    i = index_of_in(indexes_from_to, indexes_to)
    Kfrom_aug = ns.concatenate([ns.concatenate([Kfrom, np.zeros([1, ksize])], axis=0), np.zeros([ksize+1, 1])], axis=1)
    Kfull = Kfrom_aug[i][:, i]
    alphafull = noisevars
    return Kfull, alphafull

def contract_matrix(Kfull, alphafull, Fopt, indexes_from_to, indexes_to):
    i = np.array(index_of_in(indexes_from_to, indexes_to))
    i = i[i >= 0]
    Kfrom = Kfull[i][:, i]
    alpha = alphafull[i]
    Foptout = Fopt[i]
    return Kfrom, alpha, Foptout

def C_to_AB(C, ns=nt.NumpyLinalg):
    """
    C [M, T]
    """
    Cinv = 1.0 / C
    Cinvsum = ns.sum(Cinv, axis=0)
    #print "C", C
    #print "Cinvsum", Cinvsum
    B = 1.0 / Cinvsum
    A = Cinv * B
    #A[np.isnan(A)] = 0 
    return A, B

def combine_PoE_kernel(Ks, A, ns=nt.NumpyLinalg):
    M = len(Ks)
    Kpoe = 0
    for i in range(M):
        Aidiag = ns.diag(A[i])
        Kpoe += Aidiag.dot(Ks[i]).dot(Aidiag)
    return Kpoe

class Dataset(object):
    def __init__(self, lvm_kern_type, dyn_kern_type, dynamics_order=2, observed_mean_mode=ObservedMeanMode.per_part, estimation_mode=EstimationMode.ELBO, ns=nt.NumpyLinalg):
        self.estimation_mode = estimation_mode 
        self.dynamics_order = dynamics_order
        self.observed_mean_mode = observed_mean_mode
        self.ns = ns
        self.part_types = []  # list of PartType, provided
        self.mp_types = []  # list of MPType, provided
        self.lvm_kern_type = lvm_kern_type
        self.dyn_kern_type = dyn_kern_type
        self.trials = []
        self.pieces = []
        self.part_influence = {}  # part_name -> MPToPartInfluence
        self.mp_influence = {}  #  MP_name -> MPToMPInfluence
        self.trials_offsets = []
        self.y = None  # [W][N*D] observed
        self.y_means = None  # [W][D]
        self.y_centered = None  # [W][N*D]
        self.x_means = None  # [W][N*Q] latents means
        self.x_covars_diags = None  # [W][N*Q] latents diagonal covars
        self.elbo_mode = None
        self.elbo = None
        self.neg_elbo = None
        # Log-likelihood related variables
        self.loglikelihood = None
        self.neg_loglikelihood = None
        self.x_in_indexes = None  # [N, order] self.X[self.Xinindexes]
        self.x_out_indexes = None  # [N] self.X[self.Xoutindexes]
        self.ll_pp_mps = None  # (M)  list of MPs the posterior predictive was calculated for
        self.ll_pp_gps = None  # (M)(M)  list of lists of posterior predictive GPs
        self.ll_pp_couplings = None  # [M, M]  posterior predictive alphas

    def create_part_type(self, name, y_indexes, Q, M):
        part_type = PartType(name, self, index=len(self.part_types), y_indexes=y_indexes, Q=Q, M=M, ns=self.ns)
        self.part_types.append(part_type)
        return part_type

    def create_mp_type(self, name, part_type, remove_trend=False):
        mp_type = MPType(name, part_type, remove_trend=remove_trend, ns=self.ns)
        self.mp_types.append(mp_type)
        part_type.mp_types.append(mp_type)
        return mp_type

    def create_trial(self, Y):
        trial = Trial(dataset=self, Y=Y, index=len(self.trials), ns=self.ns)
        self.trials.append(trial)
        self.trials_offsets = np.cumsum([0] + [tr.npoints for tr in self.trials[:-1]]).tolist()
        return trial

    def remove_irrelevant_data(self):
        for trial in self.trials:
            trial.remove_irrelevant_data()
        self.trials_offsets = np.cumsum([0] + [tr.npoints for tr in self.trials[:-1]]).tolist()
        
    def remove_trends_in_segments(self):
        for trial in self.trials:
            trial.remove_trends_in_segments()
        
    def init_pieces(self):
        #self.remove_irrelevant_data()
        #self.remove_trends_in_segments()
        Y = np.vstack([trial.Y for trial in self.trials])
        #plx.plot_sequences(Y)
        self.npoints = Y.shape[0]
        self.y = [Y[:, part_type.y_indexes] for part_type in self.part_types]  # [W][N*D] observed
        if self.observed_mean_mode == ObservedMeanMode.per_part:
            self.y_means = [np.mean(self.y[part_type.index][part_type.get_ds_ind(), :], axis=0) for part_type in self.part_types]  # [W][D] observed
            for i, part_type in enumerate(self.part_types):
                for trial in self.trials:
                    for segment in trial.mp_segments:
                        segment.y_mean = self.y_means[i]
        elif self.observed_mean_mode == ObservedMeanMode.per_segment:
            self.y_means = [yi.copy() for yi in self.y]
            for i, part_type in enumerate(self.part_types):
                for trial in self.trials:
                    for segment in trial.mp_segments:
                        if segment.mp_type.part_type.index == i:
                            ind_t = segment.get_ds_ind()
                            mean = np.mean(self.y[i][ind_t, :], axis=0)
                            self.y_means[i][ind_t, :] = mean
                            segment.y_mean = mean
        self.y_centered = [self.ns.matrix("y_centered[{}]".format(i), self.y[i] - self.y_means[i], tags=(VarTag.observed)) for i, part_type in enumerate(self.part_types)]  # [W][N*D]
        
        # Create all pieces
        for trial in self.trials:
            trial.create_pieces()
            self.pieces.extend(trial.pieces)

        for piece in self.pieces:
            # Create mp->mp influences (couplings)
            for mp1 in piece.mps:
                for mp2 in piece.mps:
                    key = MPToMPInfluence.make_key(mp1, mp2)
                    if key not in self.mp_influence:
                        self.mp_influence[key] = MPToMPInfluence(mp1, mp2, ns=self.ns)
            # Create mp->part influences (augmenting points)
            for mp in piece.mps:
                for part_type in self.part_types:
                    key = MPToPartInfluence.make_key(mp, part_type)
                    if key not in self.part_influence:
                        print(part_type.name, part_type.Q)
                        dyn_kern = self.dyn_kern_type(ndims=self.dynamics_order*mp.part_type.Q, kern_width=1.0, suffix="["+key+"]", ns=self.ns)
                        self.part_influence[key] = MPToPartInfluence(mp, part_type, dyn_kern, ns=self.ns)
      
    def update_data_mean(self):
        # Every MPType has its own explicit mean
        for mp_type in self.mp_types:
            mp_type.update_data_mean(self)
                                              
    def init_x(self):
        # Init X with PCA
        self.x_means = []
        self.x_covars_diags = []
        for part_type in self.part_types:
            i = part_type.index
            ds_ind = part_type.get_ds_ind()
            if self.estimation_mode == EstimationMode.MAP:
                ds_ind = np.sort(list(set(ds_ind.tolist())))
                # Include up to -2 indexes more
                #ds_ind = np.sort(list(set(ds_ind.tolist()) | set((ds_ind-1).tolist()) | set((ds_ind-2).tolist())))
            #print("ds_ind", ds_ind)
            y = self.ns.get_value(self.y_centered[i])
            x = np.sqrt(len(ds_ind)) * dr.PCAreduction(y[ds_ind, :], part_type.Q)
            x_centered = x - np.mean(x, axis=0)
            #print(np.mean(x_centered))
            x_init = np.zeros([self.npoints, part_type.Q])
            x_init[ds_ind, :] = x_centered
            self.x_means.append(self.ns.matrix("x_means[{}]".format(part_type.name), 
                                               x_init, 
                                               bounds=(-10.0, 10.0),
                                               tags=(VarTag.latent_x)))
            self.x_covars_diags.append(self.ns.matrix("x_covars_diags[{}]".format(part_type.name), 
                                                      0.1 + np.zeros([self.npoints, part_type.Q]), 
                                                      bounds=(1.0e-3, 2.0),
                                                      tags=(VarTag.latent_x)))

        self.update_data_mean()

        # Every piece has its own xtminus->xtplus set mappings
        for part_type in self.part_types:
            part_type.init_x()
        for piece in self.pieces:
            piece.init_x()
            
    def init_aug_z(self, dyn_M=10):
        for part_type in self.part_types:
            part_type.init_aug_z()
        for mp_to_part_influence in self.part_influence.values():
            mp_to_part_influence.init_aug_z(self, M=dyn_M)

    def init_psi_stats(self):
        for part_type in self.part_types:
            part_type.init_psi_stats()
        for piece in self.pieces:
            piece.init_psi_stats()
        
    def init_elbo(self, elbo_mode=ELBOMode.full):
        self.elbo_mode = elbo_mode
        self.elbo = 0.0
        for part_type in self.part_types:
            part_type.init_elbo()
            self.elbo += part_type.elbo
        for piece in self.pieces:
            piece.init_elbo(elbo_mode)
            self.elbo += piece.elbo
        self.neg_elbo = -self.elbo

    def get_elbo_value(self):
        return self.ns.evaluate(self.elbo)

    def init_loglikelihood(self):
        M = len(self.part_types)  # number of parts
        # [N, order] self.X[self.Xinindexes]
        self.x_in_indexes = np.vstack([np.hstack([np.hstack([[-1] * i, trial.offset() + np.arange(trial.npoints - i)]) for trial in self.trials]) 
                        for i in range(1, self.dynamics_order+1)]).T
        # [N] self.X[self.Xoutindexes]
        self.x_out_indexes = np.hstack([trial.offset() + np.arange(trial.npoints) for trial in self.trials])
 
        for part in self.part_types:
            part.all_map_indexes = part.get_ds_ind()
 
        for mp in self.part_influence.values():
            mp.ll_set_data(self)
        
        self.loglikelihood = 0
        for part_to in self.part_types:
            indexes_to = part_to.all_map_indexes  # all marked data
            nmaps = len(indexes_to)  # number of marked mappings
            C = []  # np.Inf * np.ones([1, nmaps], dtype=np.float)  # [nMPs*M, nmaps] array of alphas, inf as default 
            Ks = []  # np.zeros  # nMPs*M*[nmaps, nmaps]  all expanden kernels from all parts and all MPs separately
            for mp_from in self.mp_types:
                indexes_from = mp_from.get_ds_ind(self)  # all mapping indexes for the current MP
                indexes_from_to = intersect(indexes_from, indexes_to)  # mapping indexes of the current MP which influence part_to outputs
                gpkey = MPToPartInfluence.make_key(mp_from, part_to)                    
                gp = self.part_influence[gpkey]
                
                noisevars = [np.Inf] * self.y[0].shape[0]  # coupling variance, inf means no influence
                for mp_to in part_to.mp_types:
                    couplingkey = MPToMPInfluence.make_key(mp_from, mp_to)
                    if couplingkey in self.mp_influence: 
                        coupling = self.mp_influence[couplingkey]
                        for i in intersect(mp_from.get_ds_ind(self), mp_to.get_ds_ind(self)):
                            noisevars[i] = coupling.alpha
                noisevars = [v for i, v in enumerate(noisevars) if i in indexes_to]
                noisevars = self.ns.stack(noisevars) 
 
                Kfull, alphafull = expand_kernel(len(gp.map_indexes), gp.ll_K, noisevars, indexes_from_to, indexes_to, ns=self.ns)
                C.append(alphafull)
                Ks.append(Kfull)
                #gp.ll_alphafull = alphafull
                #gp.ll_Kfull = Kfull

            C = self.ns.stack(C)
            #print "C", C
            A, B = C_to_AB(C, ns=self.ns)
            #print "A", A 
            #print "B", B
            #print "Ks", Ks
            Kpoe = combine_PoE_kernel(Ks, A, ns=self.ns)
            Xout = self.x_means[part_to.index][indexes_to]
            self.loglikelihood += vceq.GPDM.loglikelihood_single(Kxx=Kpoe, Xout=Xout, alpha=B, ns=self.ns)

        for part in self.part_types:
            part.init_loglikelihood()
            self.loglikelihood += part.loglikelihood
        self.neg_loglikelihood = -self.loglikelihood

    def precalc_posterior_predictive(self, mps=None):
        if self.estimation_mode == EstimationMode.ELBO:
            self.precalc_posterior_predictive_ELBO()
        else:
            self.precalc_posterior_predictive_MAP(mps)

    def precalc_posterior_predictive_ELBO(self):
        for part_type in self.part_types:
            part_type.precalc_posterior_predictive()
        for mp_type in self.mp_types:
            mp_type.precalc_posterior_predictive()
        for piece in self.pieces:
            piece.precalc_posterior_predictive()
        for pi in self.part_influence.values():
            pi.precalc_posterior_predictive()

    def precalc_posterior_predictive_MAP(self, mps):
        self.ll_pp_mps = mps
        M = len(self.part_types)  # number of parts
        # Dynamics part
        self.pp_dyn_gpfs = matrix_of_nones(M, M)
        self.ll_pp_couplings = np.zeros([M, M]) 
        for part_to in self.part_types:
            indexes_to = part_to.all_map_indexes
            mp_to = mps[part_to.index]
            C = []  # np.Inf * np.ones([1, nmaps], dtype=np.float)  # [nMPs*M, nmaps] array of alphas, inf as default 
            Ks = []  # np.zeros  # nMPs*M*[nmaps, nmaps]  all expanden kernels from all parts and all MPs separately
            for mp_from in mps:
                indexes_from = mp_from.get_ds_ind(self)
                indexes_from_to = intersect(indexes_from, indexes_to)
                noisevars = np.array([np.Inf] * self.y[0].shape[0])
                gpkey =  MPToPartInfluence.make_key(mp_from, part_to) 
                gp = self.part_influence[gpkey]
                couplingkey = MPToMPInfluence.make_key(mp_from, mp_to)
                coupling = self.mp_influence[couplingkey]
                couplingvalue = self.ns.get_value(coupling.alpha)
                print "couplingvalue", couplingvalue
                noisevars[intersect(mp_from.get_ds_ind(self), mp_to.get_ds_ind(self))] = couplingvalue
                noisevars = noisevars[part_to.all_map_indexes]
                Kfull, alphafull = expand_kernel(len(gp.map_indexes), 
                                                 self.ns.evaluate(gp.ll_K), 
                                                 noisevars, 
                                                 indexes_from_to, 
                                                 indexes_to)
                C.append(alphafull)
                Ks.append(Kfull)
            Ai, Bi = C_to_AB(np.stack(C))
            Ai[np.isnan(Ai)] = 0.0
            Xout = self.ns.get_value(self.x_means[part_to.index])[indexes_to]
            #print "Kfull", Kfull
            #print "Kfull.shape", Kfull.shape
            #print "Xout", Xout
            #print "Ci", C
            #print "Ai", Ai
            #print "Bi", Bi
            Fopt = vceq.GPDM.F_optimal_coupled_one_part(Kxxs=Ks, Xouts=Xout, Ai=Ai, Bi=Bi)
            #print "Fopt", Fopt

            i = part_to.index
            for mp_from in mps:
                part_from = mp_from.part_type 
                j = part_from.index
                indexes_from = mp_from.get_ds_ind(self)
                indexes_from_to = intersect(indexes_from, indexes_to)  # intersection of indexes where the mapping is constrained by the data
                Kfrom, alpha, Foptout = contract_matrix(Ks[j], C[j], Fopt[j], indexes_from_to, indexes_to)
                gpkey =  MPToPartInfluence.make_key(mp_from, part_to) 
                gp = self.part_influence[gpkey]
                Xin = np.concatenate([self.ns.get_value(self.x_means[j])[self.x_in_indexes[indexes_from_to, k]] for k in range(self.x_in_indexes.shape[1])], axis=1)
                self.pp_dyn_gpfs[j][i] = gpr.GP(gp.dyn_kern, 
                                                 Dx=part_from.Q*self.dynamics_order, 
                                                 Dy=part_to.Q).sample_function()
                #print "self.x_means[j]", self.x_means[j]
                #print "indexes_from_to", indexes_from_to
                #print "Xin", Xin
                #print "Foptout", Foptout
                self.pp_dyn_gpfs[j][i].condition_on(X=Xin, Y=Foptout)

                couplingkey = MPToMPInfluence.make_key(mp_from, mp_to)
                coupling = self.mp_influence[couplingkey]
                self.ll_pp_couplings[j, i] = self.ns.get_value(coupling.alpha)
        
        # LVM part
        for part_type in self.part_types:
            part_type.ll_precalc_posterior_predictive()

    def run_generative_dynamics(self, mps, nsteps, startpoint=None, start_pice_id=None):
        if self.estimation_mode == EstimationMode.ELBO:
            return self.run_generative_dynamics_ELBO(mps, nsteps, startpoint, start_pice_id)
        else:
            return self.run_generative_dynamics_MAP(mps, nsteps, startpoint)

    def run_generative_dynamics_MAP(self, mps, nsteps, startpoint=None):
        if startpoint is None:
            startpoint = self.get_dynamics_start_point_MAP()
        x_path = vceq.GPDM.generate_mean_prediction_coupled(gpfs=self.pp_dyn_gpfs,
                                                            alpha=self.ll_pp_couplings, 
                                                            x0s=startpoint, 
                                                            T=nsteps, 
                                                            order=self.dynamics_order)
        return x_path
        
    def lvm_map_to_observed_MAP(self, x):
        M = len(self.part_types)
        y = list_of_nones(M)
        for part in self.part_types:
            i = part.index 
            y[i] = part.ll_map_to_observed(x[i])  
        return y 

    def get_dynamics_start_point(self, piece, timepoint=3):
        if self.estimation_mode == EstimationMode.ELBO:
            return self.get_dynamics_start_point_ELBO(piece, timepoint)
        else:
            return self.get_dynamics_start_point_MAP(timepoint)

    def get_dynamics_start_point_MAP(self, timepoint=3):
        M = len(self.part_types)
        x_path_start = list_of_nones(M)  # W*[nsteps*Q]
        for part in self.part_types:
            i = part.index
            x_path_start[i] = self.ns.get_value(self.x_means[i])[self.x_out_indexes[timepoint-2:timepoint]]
        return x_path_start

    def find_piece_with_mps(self, mps):
        print("Searching", [mp.name for mp in mps])
        W = len(self.part_types)
        assert W == len(mps)
        for piece in self.pieces:
            print("piece.mps", piece.index, [mp.name for mp in piece.mps])
            if set(piece.mps) == set(mps):
                print("Found piece.mps", piece.index, [mp.name for mp in piece.mps])
                return piece
        return None

    def get_dynamics_start_point_ELBO(self, piece, timepoint=0, start_chunk_id=None):
        W = len(self.part_types)
        x_path_start = list_of_nones(W)  # W*[nsteps*Q]
     
        shift = 0
        if start_chunk_id is not None:
            for i in range(start_chunk_id):
                shift += piece.starts_ends[i][1] - piece.starts_ends[i][0] - 2
        
        timepoint += shift 
        # Make starting points for all parts
        for j, mp_i in zip(range(W), piece.mps):
            i = mp_i.part_type.index
            x_path_start[i] = piece.x_means_centered[i][piece.ds_xtminus_indexes[i][timepoint: timepoint+2], :] + mp_i.pp_data_mean
        print(np.hstack(x_path_start))
        return x_path_start

    def run_generative_dynamics_ELBO(self, mps, nsteps, startpoint=None, start_chunk_id=None):
        W = len(self.part_types)
        assert W == len(mps)
        piece = self.find_piece_with_mps(mps)  # template piece with all relevant posterior-predictive parameters
        if startpoint is None:
            startpoint = self.get_dynamics_start_point_ELBO(piece, start_chunk_id=start_chunk_id)
        x_path = list_of_nones(W)  # W*[nsteps*Q]
        
        # Make starting points for all parts
        for i in range(W):
            x_path[i] = 0.0 * np.ones((nsteps, self.part_types[i].Q))
            x_path[i][0:2, :] = startpoint[i]

        # Iterate nsteps-2 times
        for t in range(2, nsteps):
            x_prev = [x_path[i][t-2:t, :] for i in range(W)]
            x_t = self.run_generative_dynamics_one_step(x_prev, piece)
            for i in range(W):
                x_path[i][t, :] = x_t[i]
        return x_path

    def run_generative_dynamics_one_step(self, x_prev, piece):
        W = len(self.part_types)
        x_t = list_of_nones(W)
        alpha_inv = 1.0 / piece.pp_alpha  # [W*W]
        alpha_joint, alpha_joint_inv = vceq.GPDM.combined_PoE_uncertainty(piece.pp_alpha, ns=nt.NumpyLinalg)
        for mp_i in piece.mps:
            i = mp_i.part_type.index
            x_mean_i = mp_i.pp_data_mean
            x_ti = 0.0
            for mp_j in piece.mps:
                j = mp_j.part_type.index
                x_mean_j = mp_j.pp_data_mean
                ji_part_influence = piece.pp_mp_to_part_influences[j][i]  # !!!
                xminust_centered = np.hstack((x_prev[j][0, :] - x_mean_j, x_prev[j][1, :] - x_mean_j))
                K_xminust_z = ji_part_influence.dyn_kern.gram_matrix(xminust_centered, ji_part_influence.pp_aug_z_centered, ns=nt.NumpyLinalg)
                x_ti += alpha_inv[j, i] * K_xminust_z.dot(piece.pp_dyn_Kauginv[j][i]).dot(piece.pp_dyn_aug_u_means_centered[j][i])
            x_t[i] = alpha_joint[i] * x_ti + x_mean_i
        return x_t

    def lvm_map_to_observed(self, mps, x):
        if self.estimation_mode == EstimationMode.ELBO:
            return self.lvm_map_to_observed_ELBO(mps, x)
        else:
            return self.lvm_map_to_observed_MAP(x)

    def lvm_map_to_observed_ELBO(self, mps, x):
        W = len(self.part_types)
        piece = self.find_piece_with_mps(mps)
        y = list_of_nones(W)
        segments = piece.get_segments()
        for part in self.part_types:
            #print("Mapping part: {}".format(part.name))
            i = part.index
            K_x_z = part.lvm_kernel_obj.gram_matrix(x[i], part.pp_lvm_aug_z, ns=nt.NumpyLinalg)
            #for segment in segments:
            #    if segment.mp_type.part_type.index == i:
            #        mean_i = segment.y_mean
            #        print(segment.mp_type.part_type.name)
            #        print("segment mean[{}]={}".format(i, mean_i))
            #        print("y_means[{}]={}".format(i, self.y_means[i]))
            mean_i = self.y_means[i]
            y[i] = K_x_z.dot(part.pp_lvm_Kauginv).dot(part.pp_lvm_aug_u_means) + mean_i
        return y 
    
    def optimize_by_tags(self, func, tags=None, maxiter=None, print_vars=False):
        assert tags is not None
        if maxiter is None:
            maxiter = 100

        stored_non_differentiable = self.ns.non_differentiable
        self.ns.non_differentiable = set([var.symbol for var in self.ns.vars.values() if not tags & var.tags])
        f_df = self.ns.make_function_and_gradient(func, args=all)
        try:
            xOpt, f, d = opt.theano_optimize_bfgs_l(f_df, varargs=[opt.MaxIterations(maxiter)])
            print("Evaluated function = {}".format(self.ns.evaluate(func)))
        except np.linalg.LinAlgError as err:
            print("Exception while optimizing: {}".format(err.message))
            pass
        self.ns.non_differentiable = stored_non_differentiable
        if print_vars:
            self.print_by_tags(tags)
        return xOpt, f, d

    def save_state_to(self, filename):
        print("Writing state to {}".format(filename))
        with open(filename, "wb") as filehandle:
            cPickle.dump(self.ns.get_vars_state(), filehandle, protocol=cPickle.HIGHEST_PROTOCOL)


    def load_state_from(self, filename):
        print("Reading state from {}".format(filename))
        with open(filename, "rb") as filehandle:
            vars_state = cPickle.load(filehandle)  #, encoding='latin1')
        self.ns.set_vars_state(vars_state)


    def print_by_tags(self, tags=None):
        assert tags is not None
        for var in self.ns.vars.values():
            if tags & var.tags:
                print(var.get_name())
                print(self.ns.get_value(var))


def optimize_blocked(model, niterations=3, maxiter=300, print_vars=False, save_directory=None):
    if model.estimation_mode == EstimationMode.ELBO:
        optimize_blocked_ELBO(model, niterations, maxiter, print_vars, save_directory)
    else:
        optimize_blocked_MAP(model, niterations, maxiter, print_vars, save_directory)

def optimize_blocked_ELBO(model, niterations=3, maxiter=300, print_vars=False, save_directory=None,
        optimize_augmenting_inputs_first=True):

    if model.elbo_mode == ELBOMode.full or model.elbo_mode == ELBOMode.separate_dynamics:
        # For some reason optimizing latent inputs first helps with matrix singularities
        tags = set([VarTag.couplings, VarTag.kernel_params])
        if optimize_augmenting_inputs_first:
            tags.add(VarTag.augmenting_inputs)
        xOpt0, f0, d0 = model.optimize_by_tags(model.neg_elbo, tags=tags, maxiter=maxiter, print_vars=print_vars)

        for i in range(niterations):
            xOpt, f, d = model.optimize_by_tags(model.neg_elbo, tags=set([VarTag.latent_x]), maxiter=maxiter, print_vars=False)
            model.update_data_mean()
            xOpt, f, d = model.optimize_by_tags(model.neg_elbo, tags=set([VarTag.couplings, VarTag.kernel_params]), maxiter=maxiter, print_vars=print_vars)
            xOpt, f, d = model.optimize_by_tags(model.neg_elbo, tags=set([VarTag.augmenting_inputs]), maxiter=maxiter, print_vars=False)
            xOpt, f, d = model.optimize_by_tags(model.neg_elbo, tags=set([VarTag.couplings, VarTag.kernel_params]), maxiter=maxiter, print_vars=print_vars)
            model.precalc_posterior_predictive()
            if save_directory is not None:
                save_plot_latent_space(model, save_directory, prefix="iter_{}".format(i))
            if np.allclose(f, f0):
                return
            else:
                xOpt0, f0, d0 = xOpt, f, d

    elif model.elbo_mode == ELBOMode.couplings_only:
        model.optimize_by_tags(model.neg_elbo, tags=set([VarTag.couplings]), maxiter=maxiter, print_vars=print_vars)


def optimize_blocked_MAP(model, niterations=3, maxiter=300, print_vars=False, save_directory=None):
    # Kernel parameters and couplings must be optimized together
    model.optimize_by_tags(model.neg_loglikelihood, tags=set([VarTag.couplings, VarTag.kernel_params]), maxiter=maxiter, print_vars=print_vars)
    for i in range(niterations):
        model.optimize_by_tags(model.neg_loglikelihood, tags=set([VarTag.latent_x]), maxiter=maxiter, print_vars=False)
        model.optimize_by_tags(model.neg_loglikelihood, tags=set([VarTag.couplings, VarTag.kernel_params]), maxiter=maxiter, print_vars=print_vars)
        #model.ll_precalc_posterior_predictive()
        #if save_directory is not None:
        #    save_plot_latent_space(model, save_directory, prefix="iter_{}".format(i))
            #save_plot_latent_vs_generated(model, save_directory, prefix="iter_{}".format(i))
            #save_plot_training_vs_generated(model, save_directory, prefix="iter_{}".format(i))



##########################################################################################################
##########################################################################################################
#########################################################################################################


def save_plot_latent_space(model, directory=None, prefix=None):
    if directory is None:
        directory = "."
    if prefix is None:
        prefix = "model"
    
    print("Saving the latent space plots.")
    with open("{}/{}_alpha.txt".format(directory, prefix), 'w') as f:
        f.write("ELBO: \n")
        f.write(str(model.get_elbo_value()))
        f.write("\n")
        f.write("Couplings alpha: \n")
        for mp_influence in model.mp_influence.values():
            f.write("{} = {}".format(mp_influence.get_key(), model.ns.get_value(mp_influence.alpha)))
            f.write("\n")
        
    for mp_to_part_influence in model.part_influence.values():
        key = mp_to_part_influence.get_key()
        dyn_aug_z = model.ns.get_value(mp_to_part_influence.aug_z)
        lvm_aug_z = model.ns.get_value(mp_to_part_influence.mp_type.part_type.lvm_aug_z)
        x_means = mp_to_part_influence.mp_type.get_x_means(model)
        fig = plt.figure(figsize=(5, 5))
        plt.plot(x_means[:, 0], x_means[:, 1], '-', alpha=0.2)
        plt.plot(dyn_aug_z[:, 0], dyn_aug_z[:, 1], 'o', markersize=15, markeredgewidth=2, fillstyle="none")
        plt.plot(lvm_aug_z[:, 0], lvm_aug_z[:, 1], '+', markersize=15, markeredgewidth=2, fillstyle="none")
        plt.savefig("{}/{}_{}.pdf".format(directory, prefix, key))
        plt.close(fig)

   


def save_plot_latent_vs_generated(model, mps, directory=None, prefix=None):
    if directory is None:
        directory = "."
    if prefix is None:
        prefix = "model"

    print("Saving the latent space plots, latent vs generated.")
    N = 200
    W = len(model.part_types)  # number of parts
    x_path = model.run_generative_dynamics(mps, N)
    piece = model.find_piece_with_mps(mps)
    for mp_i in mps:
        i = mp_i.part_type.index
        x_means_i = model.ns.evaluate(piece.xtplus_centered[i]) + mp_i.pp_data_mean
        
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x_means_i, '-', alpha=0.2)
        plt.plot(x_path[i])
        plt.subplot(1, 2, 2)
        plt.plot(x_means_i[:, 0], x_means_i[:, 1], '-', alpha=0.2)
        plt.plot(x_path[i][:, 0], x_path[i][:, 1])
        
        plt.savefig("{}/{}_latent_vs_generated_part_{}.pdf".format(directory, prefix, i))
        plt.close(fig)


def save_plot_latent_vs_generated_selected_dims(model, mpss, colors, dim0=0, dim1=1, directory=None, prefix=None):
    if directory is None:
        directory = "."
    if prefix is None:
        prefix = "model"

    print("Saving the latent space plots, latent vs generated.")
    N = 200
    W = len(model.part_types)  # number of parts
    x_paths = [model.run_generative_dynamics(mps, N) for mps in mpss]
    for part in model.part_types:
        part_i = part.index
        fig = plt.figure(figsize=(5.2, 5))
        for mps_i, mps in enumerate(mpss):
            piece = model.find_piece_with_mps(mps)
            for mp in mps:
                if mp.part_type == part:
                    for start, end in piece.starts_ends:
                        x_means = model.ns.get_value(model.x_means[part_i])[start+piece.offset():end+piece.offset(), :]
                        p1, = plt.plot(x_means[:, dim0], x_means[:, dim1], '--', color=colors[mps_i][part_i], alpha=0.3)
                    p2, = plt.plot(x_paths[mps_i][part_i][:, dim0], 
                                   x_paths[mps_i][part_i][:, dim1], 
                                   color=colors[mps_i][part_i],
                                   label="{}".format(mp.name))
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.xlabel("dim {}".format(dim0))
        plt.ylabel("dim {}".format(dim1))
        plt.title("Body part: {}".format(part.name))
        plt.savefig("{}/{}_latent_vs_generated_part_{}_dims_{}-{}.pdf".format(directory, prefix, part.name, dim0, dim1))
        plt.close(fig)


def save_plot_training_vs_generated(model, mps, directory=None, prefix=None):
    if directory is None:
        directory = "."
    if prefix is None:
        prefix = "model"

    print("Saving the observed space plots, training vs generated.")
    N = 200
    W = len(model.part_types)  # number of parts
    x_path = model.run_generative_dynamics(mps, N)
    y_path = model.lvm_map_to_observed(mps, x_path)
    for i, mp in enumerate(mps):
        fig = plt.figure(figsize=(5, 5))
        y_i = mp.get_y(model)[:N, :]
        plt.plot(y_i, '-', alpha=0.2)
        plt.plot(x_path[i], "o")
        plt.plot(y_path[i])
        plt.savefig("{}/{}_training_vs_generated_part_{}.pdf".format(directory, prefix, i))
        plt.close(fig)


def save_plot_piece_posterior_predictive(model, mps, directory):
     piece = model.find_piece_with_mps(mps)
     piece.save_plot_piece_posterior_predictive(directory)
