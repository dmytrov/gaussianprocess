import numpy as np
import matplotlibex as plx
import ml.gptheano.vecgpdm.multiprimitivemodel as mdl
import ml.gptheano.vecgpdm.kernels as krn
import numerical.numpytheano.theanopool as tp
import ml.gptheano.vecgpdm.equations as vceq
import numerical.numpytheano as nt
import matplotlibex.mlplot as plx
import bvhrwroutines.bvhrwroutines as br
import ml.gptheano.vecgpdm.enums


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
    dir_out = "../../../../log/multiprimitive"
    #ns = tp.NumpyVarPool()
    ns = tp.TheanoVarPool()
    lvm_kern_type = krn.RBF_ARD_Kernel
    dyn_kern_type = krn.RBF_ARD_Kernel_noscale
    load_motion = load_bvh_motion_4
    dataset = mdl.Dataset(lvm_kern_type=lvm_kern_type, 
                          dyn_kern_type=dyn_kern_type, 
                          observed_mean_mode=mdl.ObservedMeanMode.per_part,
                          estimation_mode=EstimationMode.ELBO, ns=ns)
    
    Q = 3  # latent space dims
    lvm_M = 25
    dyn_M = 16
    motion = load_motion(dir_data + "/2016.05.03_bjoern/009_skeleton.bvh") 
    y, parts_IDs = motion.get_all_parts_data_and_IDs()
    #print("y", y.shape)
    part_indexes = br.IDs_to_indexes(parts_IDs)
    #pt_head = dataset.create_part_type("head", y_indexes=part_indexes[0], Q=Q, M=lvm_M)
    pt_left = dataset.create_part_type("left", y_indexes=part_indexes[0], Q=Q, M=lvm_M)
    pt_right = dataset.create_part_type("right", y_indexes=part_indexes[1], Q=Q, M=lvm_M)
    pt_lower = dataset.create_part_type("lower", y_indexes=part_indexes[2], Q=Q, M=lvm_M)
    #mpt_head_walk = dataset.create_mp_type("walk", pt_head)
    mpt_left_walk = dataset.create_mp_type("walk", pt_left)
    mpt_left_wave = dataset.create_mp_type("wave", pt_left)
    mpt_right_walk = dataset.create_mp_type("walk", pt_right)
    mpt_right_wave = dataset.create_mp_type("wave", pt_right)
    mpt_lower_walk = dataset.create_mp_type("walk", pt_lower)

    #motion = load_motion("./2016.05.03_bjoern/004_skeleton.bvh")  # walking side-to-wall
    #y, parts_IDs = motion.get_all_parts_data_and_IDs()
    #trial = dataset.create_trial(Y=y, learning_mode=mdl.LearningMode.full)
    #mps = [mpt_head_walk, mpt_left_walk, mpt_right_walk, mpt_lower_walk]
    #add_segments(trial, mps=mps, 
    #             starts_ends=[[45, 85],
    #                          [165, 200],
    #                          [290, 340],
    #                          [430, 475],
    #                          [585, 620],
    #                         ])
    
    motion = load_motion(dir_data + "/2016.05.03_bjoern/009_skeleton.bvh")  # walking and right-arm-waving, side-to-wall
    y, parts_IDs = motion.get_all_parts_data_and_IDs()
    trial = dataset.create_trial(Y=y, learning_mode=mdl.LearningMode.full)
    mps = [mpt_left_walk, mpt_right_wave, mpt_lower_walk]
    add_segments(trial, mps=mps, 
                 starts_ends=[#[180, 240],  # clean
                              #[320, 390],  # clean
                              #[490, 550],  # clean
                              #[670, 710],  # clean
                              [815, 860],  # clean
                             ])

    motion = load_motion(dir_data + "/2016.05.03_bjoern/010_skeleton.bvh")  # walking and left-arm-waving, side-to-wall
    y, parts_IDs = motion.get_all_parts_data_and_IDs()
    trial = dataset.create_trial(Y=y, learning_mode=mdl.LearningMode.full)
    mps = [mpt_left_wave, mpt_right_walk, mpt_lower_walk]
    add_segments(trial, mps=mps, 
                 starts_ends=[#[55, 100],  # clean
                              #[215, 260],  # clean
                              #[360, 430],  # clean
                              #[520, 580],  # clean
                              [700, 745],  # clean
                             ])

    motion = load_motion(dir_data + "/2016.05.03_bjoern/011_skeleton.bvh")  # walking and both-arms-waving, side-to-wall
    y, parts_IDs = motion.get_all_parts_data_and_IDs()
    trial = dataset.create_trial(Y=y, learning_mode=mdl.LearningMode.limited)
    mps = [mpt_left_wave, mpt_right_wave, mpt_lower_walk]
    add_segments(trial, mps=mps, 
                 starts_ends=[##[60, 95],    # lots of artefacts around
                              ##[264, 289],  # lots of artefacts around
                              ##[400, 440],  # lots of artefacts around
                              ##[600, 625],  # lots of artefacts around
                              ##[740, 774],  # lots of artefacts around
                              #[1135, 1200],  # clean
                              #[1320, 1380],  # clean
                              #[1520, 1580],  # clean
                              #[1735, 1800],  # clean
                              #[1910, 1970],  # clean
                              [2665, 2730]   # clean
                             ])


    
    #motion = load_bvh_motion_2("./2016.05.03_bjoern/003_skeleton.bvh")  # walking with the box, wall-to-wall
    #y1, parts_IDs = motion.get_all_parts_data_and_IDs()
    #part_indexes = br.IDs_to_indexes(parts_IDs)
    #pt_upper = dataset.create_part_type("upper", y_indexes=part_indexes[0], Q=3, M=8)
    #pt_lower = dataset.create_part_type("lower", y_indexes=part_indexes[1], Q=3, M=8)
    #mpt_upper_walk = dataset.create_mp_type("walk", pt_upper)
    #mpt_lower_walk = dataset.create_mp_type("walk", pt_lower)
    #trial1 = dataset.create_trial(Y=y1, learning_mode=mdl.LearningMode.full)
    #mps = [mpt_upper_walk, mpt_lower_walk]
    #add_segments(trial1, mps=mps, 
    #             starts_ends=[[80, 120],
    #                          [235, 280],
    #                          [370, 415],
    #                          [500, 530],
    #                          [635, 670],
    #                         ])

    directory = dir_out
    dataset.init_pieces()
    dataset.init_x()
    dataset.init_aug_z(dyn_M=dyn_M)
    dataset.init_psi_stats()

    step = 1
    if step == 1:
        elbo_mode = vceq.ELBOMode.separate_dynamics
        dataset.init_elbo(elbo_mode)
        print(dataset.get_elbo_value())
        dataset.precalc_posterior_predictive()
        mdl.save_plot_latent_space(dataset, directory, "initial")
        mdl.optimize_blocked(dataset, niterations=5, maxiter=300, print_vars=True, save_directory=directory)
        dataset.save_state_to(directory + "/step_1.pkl")
    elif step == 2:
        dataset.load_state_from(directory + "/step_1.pkl")
        elbo_mode = vceq.ELBOMode.couplings_only
        dataset.init_elbo(elbo_mode)
        print(dataset.get_elbo_value())
        dataset.precalc_posterior_predictive()
        mdl.optimize_blocked(dataset, maxiter=200, print_vars=True, save_directory=directory)
        dataset.save_state_to(directory + "/step_2.pkl")
    elif step == 3:
        dataset.load_state_from(directory + "/step_2.pkl")
        elbo_mode = vceq.ELBOMode.couplings_only
        dataset.init_elbo(elbo_mode)
        print(dataset.get_elbo_value())
        dataset.precalc_posterior_predictive()
        filename = directory + "/final_generated_2.bvh"
        mps = [mpt_left_wave, mpt_right_wave, mpt_lower_walk]
        mdl.save_plot_latent_space(dataset, directory, "final")
        mdl.save_plot_latent_vs_generated(dataset, mps, directory, "final")
        mdl.save_plot_training_vs_generated(dataset, mps, directory, "final")
        x_path = dataset.run_generative_dynamics(mps, nsteps=300)
        y_path = dataset.lvm_map_to_observed(mps, x_path)
        motion.set_all_parts_data(np.hstack(y_path))
        motion.write_BVH(filename)

