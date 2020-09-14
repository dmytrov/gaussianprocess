import os
from time import strftime
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlibex as plx
import ml.gptheano.vecgpdm.multiprimitivemodel as mdl
import ml.gptheano.vecgpdm.kernels as krn
import numerical.numpytheano.theanopool as tp
import ml.gptheano.vecgpdm.equations as vceq
import numerical.numpytheano as nt
import matplotlibex.mlplot as plx
import bvhrwroutines.bvhrwroutines as br
from ml.gptheano.vecgpdm.enums import *


def make_test_data_marked(N, M, trialID):
    y = np.hstack([np.vstack([[trialID] * N,
                     [m] * N,
                     range(N)
                    ]).T for m in range(M)])
    y = np.reshape(np.random.normal(size=res.size), res.shape)
    return res    


def make_test_data_sines(N, M, trialID):
    t = np.linspace(0.0, 3.1*2*np.pi, num=N)
    fs = [1.0, 3.14]
    y = np.hstack([np.vstack([5.0*np.sin(fs[i]*t + 0.0),
                              4.1*np.sin(fs[i]*t + 0.4),
                              3.1*np.sin(fs[i]*t + 1.5)]).T for i in range(M)])
    y += 0.1 * np.reshape(np.random.normal(size=y.size), y.shape)

    y = np.hstack([np.vstack([[trialID] * N, [m] * N, range(N)]).T 
            for m in range(M)])  # M * [trialID, partID, t]
    return y


def add_segments(trial, mps, starts_ends):
    nsegments = len(starts_ends)
    for start, end in starts_ends:
        for mp in mps:
            trial.add_segment(mdl.MPSegment(mp_type=mp, start=start, end=end))

if __name__ == "__main__":
    np.random.seed(1)
    ns = tp.NumpyVarPool()
    #ns = tp.TheanoVarPool()

    Ytrial0 = make_test_data_sines(N=20, M=2, trialID=0)
    Ytrial1 = -make_test_data_sines(N=30, M=2, trialID=1)
    Ytrials = np.vstack([Ytrial0, Ytrial1])
    plt.plot(Ytrials)
    plt.savefig("Ytrials.pdf")
    plt.close()
    print Ytrials
    
    lvm_kern_type = krn.ARD_RBF_Kernel
    dyn_kern_type = krn.ARD_RBF_Kernel_noscale
    dataset = mdl.Dataset(lvm_kern_type=lvm_kern_type, 
                          dyn_kern_type=dyn_kern_type, 
                          observed_mean_mode=mdl.ObservedMeanMode.per_part, 
                          estimation_mode=EstimationMode.ELBO, ns=ns)
    Q = [3, 3]  # latent space dims
    parts_IDs = [0, 0, 0, 1, 1, 1]
    part_indexes = br.IDs_to_indexes(parts_IDs)
    pt_upper = dataset.create_part_type("upper", y_indexes=part_indexes[0], Q=Q[0], M=2)
    pt_lower = dataset.create_part_type("lower", y_indexes=part_indexes[1], Q=Q[1], M=2)
    mpt_upper_walk = dataset.create_mp_type("walk", pt_upper)
    mpt_upper_wave = dataset.create_mp_type("wave", pt_upper)
    mpt_lower_walk = dataset.create_mp_type("walk", pt_lower)
    trial0 = dataset.create_trial(Y=Ytrial0)
    add_segments(trial0, mps=[mpt_upper_walk, mpt_lower_walk], starts_ends=[[5, 9], [12, 17]])
    trial1 = dataset.create_trial(Y=Ytrial1)
    add_segments(trial1, mps=[mpt_upper_wave, mpt_lower_walk], starts_ends=[[10, 16], [20, 28]])

    dataset.init_pieces()
    dataset.init_x()
    dataset.init_aug_z(dyn_M=2)
    dataset.init_psi_stats()
    dataset.init_elbo(elbo_mode=mdl.ELBOMode.full)
    dataset.init_loglikelihood()
    print("Log-likelihood = {}".format(ns.evaluate(dataset.loglikelihood)))
    #mdl.optimize_blocked(dataset, niterations=1, maxiter=50, print_vars=True)
    
    #mps = [mpt_upper_walk, mpt_lower_walk]
    #dataset.precalc_posterior_predictive(mps)
    #xstar = dataset.run_generative_dynamics(mps, nsteps=2, start_pice_id=0)
    #plt.plot(np.hstack(xstar))
    #plt.savefig("xstar(0).pdf")
    #plt.close()

    mps = [mpt_upper_wave, mpt_lower_walk]
    dataset.precalc_posterior_predictive(mps)
    xstar = dataset.run_generative_dynamics(mps, nsteps=3, start_pice_id=1)
    plt.plot(np.hstack(xstar))
    plt.savefig("xstar(1).pdf")
    plt.close()
    
    ystar = dataset.lvm_map_to_observed(mps, xstar)
    #print xstar, ystar
    plt.plot(xstar[0])
    plt.savefig("xstar[0].pdf")
    plt.close()
    
    plt.plot(ystar[0])
    plt.savefig("ystar[0].pdf")
    plt.close()
    
    plt.plot(xstar[1])
    plt.savefig("xstar[1].pdf")
    plt.close()
    
    plt.plot(ystar[1])
    plt.savefig("ystar[1].pdf")
    plt.close()
    
    
