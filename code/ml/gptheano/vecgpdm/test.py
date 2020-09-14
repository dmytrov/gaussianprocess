import os
import numpy as np
import matplotlibex as plx
import numerical.numpytheano.theanopool as tp
import ml.gptheano.vecgpdm.multiprimitivemodel as mdl
import ml.gptheano.vecgpdm.kernels as krn
import ml.gptheano.vecgpdm.equations as vceq
import numerical.numpytheano as nt
import matplotlibex.mlplot as plx

def generate_y(t, fs):
    return np.vstack([ np.vstack([5.0*np.sin(f*t+0.0), 5.0*np.sin(f*t+3.5),
                                  4.1*np.sin(f*t+0.4), 4.1*np.sin(f*t+2.8),
                                  3.1*np.sin(f*t+0.4), 3.1*np.sin(f*t+3.8),
                                  2.1*np.sin(f*t+0.4), 2.1*np.sin(f*t+4.8)]) for f in fs]).T

if __name__ == "__main__":
    t = np.linspace(0.0, 3.1*2*np.pi, num=600)
    y1 = generate_y(t, fs=[1.0, 3.14, 1.0])
    y2 = generate_y(t, fs=[1.0, 1.0, 3.14])
    y3 = generate_y(t, fs=[1.0, 3.14, 3.14])
  
    np.random.seed(0)
    y = [y1 + 0.2*np.reshape(np.random.normal(size=y1.size), y1.shape),
         y2 + 0.2*np.reshape(np.random.normal(size=y2.size), y2.shape) + 1.0,
         y3 + 0.2*np.reshape(np.random.normal(size=y3.size), y3.shape) + 2.0,
         ]
    
    #ns = tp.NumpyVarPool()
    ns = tp.TheanoVarPool()
    lvm_kern_type = krn.RBF_ARD_Kernel
    dyn_kern_type = krn.RBF_ARD_Kernel_noscale
    dataset = mdl.Dataset(lvm_kern_type=lvm_kern_type, dyn_kern_type=dyn_kern_type, observed_mean_mode=mdl.ObservedMeanMode.per_part, ns=ns)
    pt_1 = dataset.create_part_type("1", y_indexes=range(0, 8), Q=2, M=10)
    pt_2 = dataset.create_part_type("2", y_indexes=range(8, 16), Q=2, M=10)
    pt_3 = dataset.create_part_type("3", y_indexes=range(16, 24), Q=2, M=10)
    mpt_sin_1 = dataset.create_mp_type("sin", pt_1)
    mpt_sin_2 = dataset.create_mp_type("sin", pt_2)
    mpt_sin_3 = dataset.create_mp_type("sin", pt_3)
    mpt_up_1 = dataset.create_mp_type("up", pt_1)
    mpt_up_2 = dataset.create_mp_type("up", pt_2)
    mpt_up_3 = dataset.create_mp_type("up", pt_3)
    
    start = 100
    end = 500
    trial1 = dataset.create_trial(Y=y[0], learning_mode=mdl.LearningMode.full)
    trial1.add_segment(mdl.MPSegment(mp_type=mpt_sin_1, start=start, end=end))
    trial1.add_segment(mdl.MPSegment(mp_type=mpt_up_2, start=start, end=end))
    trial1.add_segment(mdl.MPSegment(mp_type=mpt_sin_3, start=start, end=end))
    
    trial2 = dataset.create_trial(Y=y[1], learning_mode=mdl.LearningMode.full)
    trial2.add_segment(mdl.MPSegment(mp_type=mpt_sin_1, start=start, end=end))
    trial2.add_segment(mdl.MPSegment(mp_type=mpt_sin_2, start=start, end=end))
    trial2.add_segment(mdl.MPSegment(mp_type=mpt_up_3, start=start, end=end))
    
    trial3 = dataset.create_trial(Y=y[2], learning_mode=mdl.LearningMode.limited)
    trial3.add_segment(mdl.MPSegment(mp_type=mpt_sin_1, start=start, end=end))
    trial3.add_segment(mdl.MPSegment(mp_type=mpt_up_2, start=start, end=end))
    trial3.add_segment(mdl.MPSegment(mp_type=mpt_up_3, start=start, end=end))
    
    dataset.init_pieces()
    dataset.init_x()
    dataset.init_aug_z(dyn_M=20)
    dataset.init_psi_stats()
    
    directory = "test"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    steps = [1, 2, 3, 4]
    for step in steps:
        if step == 1:
            elbo_mode = vceq.ELBOMode.separate_dynamics
            dataset.init_elbo(elbo_mode)
            print("ELBO: {}".format(dataset.get_elbo_value()))
            dataset.precalc_posterior_predictive()
            mdl.save_plot_latent_space(dataset, directory, "initial")
            mdl.optimize_blocked(dataset, niterations=5, maxiter=300, print_vars=True, save_directory=directory)
            dataset.save_state_to(directory + "/step_1.pkl")
        elif step == 2:
            dataset.load_state_from(directory + "/step_1.pkl")
            elbo_mode = vceq.ELBOMode.couplings_only
            dataset.init_elbo(elbo_mode)
            print("ELBO: {}".format(dataset.get_elbo_value()))
            mdl.optimize_blocked(dataset, maxiter=300, print_vars=True, save_directory=directory)
            dataset.save_state_to(directory + "/step_2.pkl")
        elif step == 3:
            dataset.load_state_from(directory + "/step_2.pkl") 
            elbo_mode = vceq.ELBOMode.couplings_only
            dataset.init_elbo(elbo_mode)
            dataset.precalc_posterior_predictive()
            #mps = [mpt_sin_1, mpt_up_2, mpt_sin_3]
            mps = [mpt_sin_1, mpt_up_2, mpt_up_3]
            x_path = dataset.run_generative_dynamics(mps, nsteps=100)
            y_path = dataset.lvm_map_to_observed(mps, x_path)
    
            mdl.save_plot_piece_posterior_predictive(dataset, mps, directory)
            mdl.save_plot_latent_space(dataset, directory, "final_3")
            mdl.save_plot_latent_vs_generated(dataset, mps, directory, "final_3")
            mdl.save_plot_training_vs_generated(dataset, mps, directory, "final_3")
        elif step == 4:
            dataset.load_state_from(directory + "/step_2.pkl") 
            elbo_mode = vceq.ELBOMode.full
            dataset.init_elbo(elbo_mode)
            dataset.precalc_posterior_predictive()
            #mps = [mpt_sin_1, mpt_sin_2, mpt_up_3]
            mps = [mpt_sin_1, mpt_up_2, mpt_sin_3]
            x_path = dataset.run_generative_dynamics(mps, nsteps=100)
            y_path = dataset.lvm_map_to_observed(mps, x_path)
    
            mdl.save_plot_piece_posterior_predictive(dataset, mps, directory)
            mdl.save_plot_latent_space(dataset, directory, "final_4")
            mdl.save_plot_latent_vs_generated(dataset, mps, directory, "final_4")
            mdl.save_plot_training_vs_generated(dataset, mps, directory, "final_4")


