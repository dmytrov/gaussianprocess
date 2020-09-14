import numpy as np
import numerical.numpyext.linalg as ntla
import numerical.numpyext as npx
import numerical.numpyext.dimreduction as dr
import theano
import ml.gp.vcgpdm.equations as vceq
import ml.gp.equations as gpeq

class PartData(object):
    def __init__(self,
                 Y, 
                 sequences_indexes): 
        self.Y = Y
        self.sequences_indexes = sequences_indexes
        self.N, self.D = self.Y.shape
        self.Y_mean = np.mean(self.Y, axis=0)
        self.Y_centered = self.Y - self.Y_mean

class PartParams(object):
    def __init__(self,
                 data,  # =PartData(),
                 Q,  # =2, latent dimensionality
                 xt0indexes,
                 xtminusindexes,
                 xtplusindexes,
                 M  # =10, number of inducing mappings
                 ):
        self.data = data
        self.Q = Q
        self.dynamics_order = 2
        self.xt0indexes, self.xtminusindexes, self.xtplusindexes = xt0indexes, xtminusindexes, xtplusindexes
        self.M = M


class ModelData(object):
    def __init__(self,
                 Y_sequences):  # =[np.ones([5, 3]), np.ones([6, 3])]):
        if not isinstance(Y_sequences, list):
            Y_sequences = [Y_sequences]
        self.Y_sequences = Y_sequences
        self.Y = np.vstack(Y_sequences)
        self.sequences_indexes = gpeq.sequences_indexes(Y_sequences)
        self.N, self.D = self.Y.shape

class ModelParams(object):
    def __init__(self,
                 data,  # =ModelData(),
                 Qs,  # =[2, 2, 2], latent dimensions
                 parts_IDs,  # =[0, 0, 1, 1, 2, 2]
                 M  # =10, number of inducing mappings
                 ):
        self.data = data
        self.Qs = Qs
        self.parts_IDs = parts_IDs
        self.dynamics_order = 2
        self.M = M

        assert len(parts_IDs) == data.D
        assert min(parts_IDs) == 0
        self.nparts = max(parts_IDs) + 1
        self.parts_indexes = [[j for j, index in enumerate(parts_IDs) if i == index] for i in range(self.nparts)]
        self.xt0indexes, self.xtminusindexes, self.xtplusindexes = gpeq.xt0_xtminus_xtplus_indexes(data.sequences_indexes, self.dynamics_order)
        self.xt0indexes, self.xtminusindexes, self.xtplusindexes = np.array(self.xt0indexes), np.array(self.xtminusindexes), np.array(self.xtplusindexes)
        #self.xt0indexes_var = IntVectorVariable("Xt0Indexes", self.xt0indexes)
        #self.xtplusindexes_var = IntVectorVariable("XtplusIndexes", self.xtplusindexes)

    def make_part_params(self, part_ID):
        res = PartParams(PartData(self.data.Y[:, self.parts_indexes[part_ID]], self.data.sequences_indexes),
                         self.Qs[part_ID],
                         self.xt0indexes,
                         self.xtminusindexes,
                         self.xtplusindexes,
                         self.M)
        return res

    def make_parts_params(self):
        return [self.make_part_params(part_ID) for part_ID in range(self.nparts)]


class VCGPDMPart(object):
    def __init__(self, 
                 partparams  # =PartParams()
                 ):
        self.params = partparams
        Q = self.params.Q
        N = self.params.data.N

        self.x_means = dr.PCAreduction(self.params.data.Y_centered, Q)
        self.x_covars = np.identity(Q)[np.newaxis, :, :] * np.ones(N)[:, np.newaxis, np.newaxis]
        
        # Dynamics
        self.dyn_kernparam_alpha = 1  # x->x mapping noise
        self.dyn_kernparam_sigmasqrf = 1  # scaling coefficient
        self.dyn_kernparam_lambda1 = 1  # for x_t-1
        self.dyn_kernparam_lambda2 = 1  # for x_t-2
        self.dyn_kernparam_lambdas = np.concatenate([np.ones(Q) * self.dyn_kernparam_lambda2, 
                                                     np.ones(Q) * self.dyn_kernparam_lambda1])
        self.dyn_xtminus_means = np.hstack([self.x_means[self.params.xtminusindexes[0], :], self.x_means[self.params.xtminusindexes[1], :]])  
        self.dyn_xtminus_covars = np.dstack([np.hstack([self.x_covars[self.params.xtminusindexes[0], :, :], np.zeros_like(self.x_covars[self.params.xtminusindexes[0], :, :])]),
                                             np.hstack([np.zeros_like(self.x_covars[self.params.xtminusindexes[1], :, :]), self.x_covars[self.params.xtminusindexes[1], :, :]])])
        self.dyn_xt_means = self.x_means[self.params.xtplusindexes, :]
        self.dyn_xt_covars = self.x_covars[self.params.xtplusindexes, :, :]
        nmappings = self.params.xtplusindexes.size
        aug_indexes = np.arange(0, nmappings, int(nmappings/self.params.M))
        self.dyn_aug_in = np.hstack([self.x_means[self.params.xtminusindexes[0, aug_indexes], :], self.x_means[self.params.xtminusindexes[1, aug_indexes], :]])  
        self.dyn_aug_out = self.x_means[self.params.xtplusindexes[aug_indexes], :]
        self.dyn_psi0 = vceq.RBF_ARD_kern.psi_stat_0(self.dyn_kernparam_sigmasqrf, 
                                                     self.dyn_xtminus_means)
        self.dyn_psi1 = None  
        self.dyn_psi2 = None  
        self.dyn_Kaug = None  
        self.dyn_Kauginv = None  
        self.dyn_Kaug_partial = vceq.RBF_ARD_kern.gram_matrix(self.dyn_kernparam_sigmasqrf, 
                                                              self.dyn_kernparam_lambdas, 
                                                              self.dyn_aug_in, 
                                                              self.dyn_aug_in)
        self.dyn_psi1_partial = vceq.RBF_ARD_kern.psi_stat_1(self.dyn_kernparam_sigmasqrf, 
                                                             self.dyn_kernparam_lambdas, 
                                                             self.dyn_aug_in,
                                                             self.dyn_xtminus_means,
                                                             self.dyn_xtminus_covars)
        self.dyn_psi2_partial = vceq.RBF_ARD_kern.psi_stat_2(self.dyn_kernparam_sigmasqrf, 
                                                             self.dyn_kernparam_lambdas, 
                                                             self.dyn_aug_in,
                                                             self.dyn_xtminus_means,
                                                             self.dyn_xtminus_covars)

        # (GP) Latent Variable Model
        nmappings = self.params.xtplusindexes.size
        aug_indexes = np.arange(0, nmappings, int(nmappings/self.params.M))
        self.lvm_aug_in = self.x_means[aug_indexes, :]
        self.lvm_aug_out = self.params.data.Y_centered[aug_indexes, :]
        self.lvm_kernparam_beta = 1  # x->y mapping noise
        self.lvm_kernparam_sigmasqrf = 1  # scaling coefficient
        self.lvm_kernparam_lambda = 1
        self.lvm_kernparam_lambdas = np.ones(Q) * self.lvm_kernparam_lambda
        self.lvm_Kaug = vceq.RBF_ARD_kern.gram_matrix(self.lvm_kernparam_sigmasqrf, 
                                                      self.lvm_kernparam_lambdas, 
                                                      self.lvm_aug_in, 
                                                      self.lvm_aug_in)
        self.lvm_Kauginv = np.linalg.inv(self.lvm_Kaug)
        self.lvm_psi0 = vceq.RBF_ARD_kern.psi_stat_0(self.lvm_kernparam_sigmasqrf, 
                                                     self.x_means)
        self.lvm_psi1 = vceq.RBF_ARD_kern.psi_stat_1(self.lvm_kernparam_sigmasqrf, 
                                                     self.lvm_kernparam_lambdas, 
                                                     self.lvm_aug_in,
                                                     self.x_means,
                                                     self.x_covars)
        self.lvm_psi2 = vceq.RBF_ARD_kern.psi_stat_2(self.lvm_kernparam_sigmasqrf, 
                                                     self.lvm_kernparam_lambdas, 
                                                     self.lvm_aug_in,
                                                     self.x_means,
                                                     self.x_covars)
    def elbo(self):
        dyn_elbo = vceq.VariationalGPDM.elbo_kron(self.dyn_xt_means, 
                                                  self.dyn_xt_covars, 
                                                  self.dyn_kernparam_alpha,
                                                  self.dyn_Kaug,
                                                  self.dyn_psi0,
                                                  self.dyn_psi1,
                                                  self.dyn_psi2)
        lvm_elbo = vceq.VariationalGPLVM.elbo(self.params.data.Y, 
                                              self.lvm_kernparam_beta, 
                                              self.lvm_Kaug, 
                                              self.lvm_psi0,
                                              self.lvm_psi1,
                                              self.lvm_psi2)
        
        x0_elbo = vceq.gaussians_cross_entropies(self.x_means[self.params.xt0indexes, :],
                                                 self.x_covars[self.params.xt0indexes, :, :],
                                                 np.zeros([self.params.xt0indexes.size, self.params.Q]), 
                                                 np.identity(self.params.Q)[np.newaxis, :, :] * np.ones(self.params.xt0indexes.size)[:, np.newaxis, np.newaxis])
        x0_elbo += vceq.gaussians_cross_entropies(self.x_means[self.params.xt0indexes+1, :], 
                                                  self.x_covars[self.params.xt0indexes+1, :, :], 
                                                  self.x_means[self.params.xt0indexes, :], 
                                                  np.identity(self.params.Q)[np.newaxis, :, :] * np.ones(self.params.xt0indexes.size)[:, np.newaxis, np.newaxis])
        x0_elbo = np.sum(x0_elbo)
        #x0_elbo = 0
        #for x0index in self.params.xt0indexes:
        #    x0_elbo += vceq.gaussians_cross_entropy(self.x_means[x0index, :], self.x_covars[x0index, :, :], 
        #                                            np.zeros(self.params.Q), np.identity(self.params.Q))
        #    x0_elbo += vceq.gaussians_cross_entropy(self.x_means[x0index+1, :], self.x_covars[x0index+1, :, :], 
        #                                            self.x_means[x0index, :], np.identity(self.params.Q))
        return dyn_elbo + lvm_elbo + x0_elbo

class VCGPDM(object):
    def __init__(self, 
                 modelparams  # =ModelParams()
                 ):
        self.params = modelparams
        W = self.params.nparts
        self.parts = [VCGPDMPart(self.params.make_part_params(part_ID)) for part_ID in range(self.params.nparts)]

        self.couplingcovars = np.ones([W, W])  # [W*W], from i-th part to j-th 
        self.noisecovars = 1.0 / np.sum(1.0 / self.couplingcovars, axis=1)  # [W], total covariance for i-th part dynamics
        self.kernelweights = self.noisecovars[:, np.newaxis]**2 / self.couplingcovars**2

        # Compute the coupled dynamics augmenting kernel matrices 
        for j in range(self.params.nparts):
            part = self.parts[j]
            part.dyn_Kaug = 0
            for i in range(self.params.nparts):
                part.dyn_Kaug += self.couplingcovars[i, j] * self.parts[i].dyn_Kaug_partial
            part.dyn_Kauginv = np.linalg.inv(part.dyn_Kaug)

        # Compute Psi_1-statistics for the dynamics
        for j in range(self.params.nparts):
            part = self.parts[j]
            part.dyn_psi1 = 0
            for i in range(self.params.nparts):
                part.dyn_psi1 += self.couplingcovars[i, j] * self.parts[i].dyn_psi1_partial
                
        # Compute Psi_2-statistics for the dynamics
        for j in range(self.params.nparts):
            part = self.parts[j]
            part.dyn_psi2 = 0
            for i in range(self.params.nparts):
                for k in range(self.params.nparts):
                    if i == k:
                        part.dyn_psi2 += self.couplingcovars[i, j]**2 * self.parts[i].dyn_psi2_partial  # as it is quadratic
                    else:
                        part.dyn_psi2 += self.couplingcovars[i, j] * self.parts[i].dyn_psi1_partial[:, np.newaxis, :] \
                                       * self.couplingcovars[k, j] * self.parts[k].dyn_psi1_partial[:, :, np.newaxis]
            
    def elbo(self):
        return np.sum([part.elbo() for part in self.parts])
