import os
from time import strftime
import numpy as np
if "DISPLAY" not in os.environ:
    import matplotlib
    print("No DISPLAY found. Switching to noninteractive matplotlib backend...")
    print("Old backend is: {}".format(matplotlib.get_backend()))
    matplotlib.use('Agg')
    print("New backend is: {}".format(matplotlib.get_backend()))
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


def load_bvh_motion_2(bvhfilename):
    motion = br.BVH_Bridge()
    motion.read_BVH(bvhfilename)
    motion.add_part("upper", ["pelvis_spine1",
                              "spine1",
                              "neck",
                              "spine2",
                              "left_manubrium",
                              "left_clavicle",
                              "left_humerus",
                              "left_radius",
                              "right_manubrium",
                              "right_clavicle",
                              "right_humerus",
                              "right_radius",
                              ])
    motion.add_part("lower", ["pelvis",
                              "pelvis_right_femur",
                              "right_femur_tibia",
                              "right_tibia_foot",
                              "pelvis_left_femur",
                              "left_femur_tibia",
                              "left_tibia_foot",
                              ])
    return motion


def load_bvh_motion_4(bvhfilename):
    motion = br.BVH_Bridge()
    motion.read_BVH(bvhfilename)
    motion.add_part("head", ["neck",
                             "spine2",
                             ])
    motion.add_part("left",  ["left_manubrium",
                              "left_clavicle",
                              "left_humerus",
                              "left_radius",
                              ])
    motion.add_part("right", ["right_manubrium",
                              "right_clavicle",
                              "right_humerus",
                              "right_radius",
                              ])
    motion.add_part("lower", ["pelvis",
                              "pelvis_right_femur",
                              "right_femur_tibia",
                              "right_tibia_foot",
                              "pelvis_left_femur",
                              "left_femur_tibia",
                              "left_tibia_foot",
                              "pelvis_spine1",
                              "spine1",
                              ])
    return motion


def load_bvh_motion_3(bvhfilename):
    motion = br.BVH_Bridge()
    motion.read_BVH(bvhfilename)
    motion.add_part("left",  ["left_manubrium",
                              "left_clavicle",
                              "left_humerus",
                              "left_radius",
                              ])
    motion.add_part("right", ["right_manubrium",
                              "right_clavicle",
                              "right_humerus",
                              "right_radius",
                              ])
    motion.add_part("lower", ["pelvis",
                              "pelvis_right_femur",
                              "right_femur_tibia",
                              "right_tibia_foot",
                              "pelvis_left_femur",
                              "left_femur_tibia",
                              "left_tibia_foot",
                              "pelvis_spine1",
                              "spine1",
                              "spine2",
                              "neck",
                              ])
    return motion


def add_segments(trial, mps, starts_ends):
    nsegments = len(starts_ends)
    for start, end in starts_ends:
        for mp in mps:
            trial.add_segment(mdl.MPSegment(mp_type=mp, start=start, end=end))

if __name__ == "__main__":
    dir_data = "../../../../data/phasespace"
    dir_out = "../../../../log/multiprimitive" + strftime("(%Y-%m-%d-%H.%M.%S)")
    dir_out = "../../../../log/multiprimitive_1"
    
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    #ns = tp.NumpyVarPool()
    ns = tp.TheanoVarPool()
    lvm_kern_type = krn.RBF_ARD_Kernel
    dyn_kern_type = krn.RBF_ARD_Kernel_noscale
    load_motion = load_bvh_motion_2
    dataset = mdl.Dataset(lvm_kern_type=lvm_kern_type, 
                          dyn_kern_type=dyn_kern_type, 
                          observed_mean_mode=mdl.ObservedMeanMode.per_part, 
                          estimation_mode=EstimationMode.ELBO, ns=ns)

    Q = [3, 3]  # latent space dims
    lvm_M = 5
    dyn_M = 5
    motion = load_motion(dir_data + "/2018.01.30_olaf/walking-01_skeleton.bvh")  # just to load the skeleton
    #motion = load_motion(dir_data + "/2016.05.03_bjoern/009_skeleton.bvh")  # just to load the skeleton    
    y, parts_IDs = motion.get_all_parts_data_and_IDs()
    part_indexes = br.IDs_to_indexes(parts_IDs)
    pt_upper = dataset.create_part_type("upper", y_indexes=part_indexes[0], Q=Q[0], M=lvm_M)
    pt_lower = dataset.create_part_type("lower", y_indexes=part_indexes[1], Q=Q[1], M=lvm_M)
    mpt_upper_walk = dataset.create_mp_type("walk", pt_upper)
    mpt_upper_wave = dataset.create_mp_type("wave", pt_upper, remove_trend=True)
    mpt_lower_walk = dataset.create_mp_type("walk", pt_lower)

    motion = load_motion(dir_data + "/2018.01.30_olaf/walking-01_skeleton.bvh")  # walking wall-to-wall
    #motion = load_motion(dir_data + "/2016.05.03_bjoern/003_skeleton.bvh")  # walking wall-to-wall
    y, parts_IDs = motion.get_all_parts_data_and_IDs()
    trial = dataset.create_trial(Y=y)
    mps = [mpt_upper_walk, mpt_lower_walk]
    add_segments(trial, mps=mps,
                 starts_ends=[  [609, 713], 
                                [1144, 1244],
                                [1672, 1774], 
                                [2245, 2342],
                                [2783, 2882],
                                [3339, 3437],
                                [3859, 3959],
                                [5622, 5722],
                                [7126, 7226],
                                [7637, 7737],
                             ])

    motion = load_motion(dir_data + "/2018.01.30_olaf/protesting_walking_2arms-01_skeleton.bvh")  # walking and both-arms-waving, side-to-wall
    #motion = load_motion(dir_data + "/2016.05.03_bjoern/011_skeleton.bvh")  # walking and both-arms-waving, side-to-wall
    y, parts_IDs = motion.get_all_parts_data_and_IDs()
    trial = dataset.create_trial(Y=y)
    mps = [mpt_upper_wave, mpt_lower_walk]
    add_segments(trial, mps=mps,
                 starts_ends=[  [664, 743],
                                [920, 1001],
                                [1209, 1287],
                                [1477, 1556],
                                [1752, 1830],
                                [2028, 2108],
                                [2294, 2372],
                                [2532, 2610],
                                [2787, 2867],
                                [3024, 3103],
                             ])


    directory = dir_out
    dataset.init_pieces()
    dataset.init_x()
    print("dataset.init_aug_z")
    dataset.init_aug_z(dyn_M=dyn_M)
    dataset.init_psi_stats()

    steps = [1, 4]
    for step in steps:
        if step == 0:
            # Plot the optimization cost vs steps
            dataset.optimizationlog.load_from(directory + "/optimizationlog.pkl")
            dataset.optimizationlog.plot(transformation=tp.select_decreasing_negate)
        if step == 1:
            elbo_mode = mdl.ELBOMode.full
            dataset.init_elbo(elbo_mode)
            #print(dataset.get_elbo_value())
            dataset.precalc_posterior_predictive()
            mdl.save_plot_latent_space(dataset, dir_out, "initial")
            mdl.optimize_blocked(dataset, niterations=5, maxiter=300, print_vars=True, save_directory=directory)
            dataset.save_state_to(directory + "/step_1.pkl")
            dataset.save_state_to(directory + "/step_2.pkl")
        elif step == 2:
            # Just load, optimize separate dynamics, and save
            dataset.load_state_from(directory + "/step_1.pkl")
            elbo_mode = mdl.ELBOMode.separate_dynamics
            dataset.init_elbo(elbo_mode)
            dataset.precalc_posterior_predictive()
            mdl.save_plot_latent_space(dataset, directory, "intermediate")
            mdl.optimize_blocked(dataset, niterations=20, maxiter=50, print_vars=True, save_directory=directory)
            dataset.save_state_to(directory + "/step_1.pkl")
        elif step == 3:
            # Just load, optimize couplings, and save
            dataset.load_state_from(directory + "/step_1.pkl")
            elbo_mode = mdl.ELBOMode.couplings_only
            dataset.init_elbo(elbo_mode)
            mdl.optimize_blocked(dataset, niterations=20, maxiter=50, print_vars=True, save_directory=directory)
            dataset.precalc_posterior_predictive()
            dataset.save_state_to(directory + "/step_2.pkl")
        elif step == 4:
            # Generate motion, make plots
            elbo_mode = mdl.ELBOMode.full
            dataset.init_elbo(elbo_mode)
            dataset.load_state_from(directory + "/step_2.pkl")
            dataset.precalc_posterior_predictive()
            mpss = [[mpt_upper_walk, mpt_lower_walk],
                    [mpt_upper_wave, mpt_lower_walk]]
            for mps in mpss:
                name = "final_({})".format(".".join([mp.name for mp in mps]))
                mdl.save_plot_latent_space(dataset, directory, name)
                mdl.save_plot_latent_vs_generated(dataset, mps, directory, name)
                mdl.save_plot_training_vs_generated(dataset, mps, directory, name)
                x_path = dataset.run_generative_dynamics(mps, nsteps=300)
                y_path = dataset.lvm_map_to_observed(mps, x_path)
                motion.set_all_parts_data(np.hstack(y_path))
                motion.write_BVH("{}/{}.bvh".format(directory, name))
        elif step == 5:
            dataset.load_state_from(directory + "/step_1.pkl")
            elbo_mode = mdl.ELBOMode.full
            dataset.init_elbo(elbo_mode)
            print(dataset.get_elbo_value())
            dataset.precalc_posterior_predictive()
            mpss = [[mpt_upper_walk, mpt_lower_walk],
                    [mpt_upper_wave, mpt_lower_walk]]
            colors = [["red", "red"],
                      ["blue", "blue"]]
            mdl.save_plot_latent_vs_generated_selected_dims(dataset, mpss, colors=colors, dim0=1, dim1=2, directory=directory, prefix="paper")
            mdl.save_plot_latent_vs_generated_selected_dims(dataset, mpss, colors=colors, dim0=0, dim1=1, directory=directory, prefix="paper")
            mdl.save_plot_latent_vs_generated_selected_dims(dataset, mpss, colors=colors, dim0=0, dim1=2, directory=directory, prefix="paper")

