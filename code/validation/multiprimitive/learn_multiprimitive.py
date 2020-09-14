import os
from time import strftime
import copy
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
from validation.common import *
import argparse


validation_skip = 5

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




def run_multiprimitive_CGPDM_crossvalidation(training, validation, settings, bvhpartitioner=None):

    def add_segments(trial, mps, starts_ends, hold_id=None):
        ses = copy.deepcopy(starts_ends)
        if hold_id is not None:
            # Put the held-out primer at the beginning
            ses[hold_id][1] = ses[hold_id][0] + settings["validation_seed_size"]
        print(ses)
        for start, end in ses:
            for mp in mps:
                trial.add_segment(mdl.MPSegment(mp_type=mp, start=start, end=end))


    hold_id = settings["hold"]
    dir_data = "../../../data/phasespace"
    dir_out = settings["directory"]
    completed_filename = dir_out + "/completed.pkl"
    if os.path.exists(completed_filename):
        return

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
    lvm_M = settings["lvm_Ms"][0]
    dyn_M = settings["dyn_Ms"][0]
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
    mps1 = [mpt_upper_walk, mpt_lower_walk]
    starts_ends=[   [609, 713], 
                    [1144, 1244],
                    [1672, 1774], 
                    [2245, 2342],
                    [2783, 2882],
                    [3339, 3437],
                    [3859, 3959],
                    [5622, 5722],
                    [7126, 7226],
                    [7637, 7737],
                ]
    add_segments(trial, mps=mps1, starts_ends=starts_ends, hold_id=hold_id)
    held_walk_walk = y[starts_ends[hold_id][0]:starts_ends[hold_id][1], :]

    motion = load_motion(dir_data + "/2018.01.30_olaf/protesting_walking_2arms-01_skeleton.bvh")  # walking and both-arms-waving, side-to-wall
    #motion = load_motion(dir_data + "/2016.05.03_bjoern/011_skeleton.bvh")  # walking and both-arms-waving, side-to-wall
    y, parts_IDs = motion.get_all_parts_data_and_IDs()
    trial = dataset.create_trial(Y=y)
    mps2 = [mpt_upper_wave, mpt_lower_walk]
    starts_ends=[   [664, 743],
                    [920, 1001],
                    [1209, 1287],
                    [1477, 1556],
                    [1752, 1830],
                    [2028, 2108],
                    [2294, 2372],
                    [2532, 2610],
                    [2787, 2867],
                    [3024, 3103],
                ]
    add_segments(trial, mps=mps2, starts_ends=starts_ends, hold_id=hold_id)
    held_wave_walk = y[starts_ends[hold_id][0]:starts_ends[hold_id][1], :]

    directory = dir_out
    dataset.init_pieces()
    dataset.init_x()
    dataset.init_aug_z(dyn_M=dyn_M)
    dataset.init_psi_stats()

    elbo_mode = mdl.ELBOMode.full
    dataset.init_elbo(elbo_mode)
    #print(dataset.get_elbo_value())
    dataset.precalc_posterior_predictive()
    mdl.save_plot_latent_space(dataset, dir_out, "initial")
    t0 = time.time()        
    if not settings["dry_run"]:
        mdl.optimize_blocked(dataset, niterations=settings["maxrun"], maxiter=settings["maxiter"], print_vars=True, save_directory=directory)
    t1 = time.time()
    dataset.save_state_to(directory + "/step_1.pkl")
    dataset.save_state_to(directory + "/step_2.pkl")

    # Generate motion, make plots
    elbo_mode = mdl.ELBOMode.full
    dataset.init_elbo(elbo_mode)
    dataset.load_state_from(directory + "/step_2.pkl")
    dataset.precalc_posterior_predictive()
    mpss = [mps1, mps2]
    held_data = [held_walk_walk, held_wave_walk]
    for i, mps in enumerate(mpss):
        name = "final_({})".format(".".join([mp.name for mp in mps]))
        mdl.save_plot_latent_space(dataset, directory, name)
        mdl.save_plot_latent_vs_generated(dataset, mps, directory, name)
        mdl.save_plot_training_vs_generated(dataset, mps, directory, name)
        x_path = dataset.run_generative_dynamics(mps, nsteps=300, start_pice_id=hold_id)
        y_path = dataset.lvm_map_to_observed(mps, x_path)
        motion.set_all_parts_data(np.hstack(y_path))
        motion.write_BVH("{}/{}.bvh".format(directory, name))

        # Compute the crossvalidation errors
        held_y = held_data[i]
        nsteps = held_y.shape[0]
        x_path = dataset.run_generative_dynamics(mps, nsteps=nsteps, start_pice_id=hold_id)
        y_path = dataset.lvm_map_to_observed(mps, x_path)
        errors = compute_errors(observed=held_y,
                        predicted=np.hstack(y_path))
        errors["ELBO"] = dataset.get_elbo_value()
        errors["timing"] = t1-t0
        errors["settings"] = settings
        # Write the errors
        with open(dir_out + "/errors_mps({}).pkl".format(i), "wb") as filehandle:
            pickle.dump(errors, filehandle)
    
    with open(completed_filename, "wb") as filehandle:
        pickle.dump("completed", filehandle)


def create_model_iterator(id=1):
    miter = ModelIterator()
    # Dummy recording
    miter.load_recording(
        ds.Recordings.exp3_walk,
        bodypart_motiontypes=[(ds.BodyParts.Upper, ds.MotionType.WALK_PHASE_ALIGNED),
                              (ds.BodyParts.Lower, ds.MotionType.WALK_PHASE_ALIGNED)],
        max_chunks=10)
    miter.settings = {"validation_seed_size": 4,  # number of frames to be included into the training set
                      "dry_run": False,  # skip optimization completely
                      "Qs": [3] * miter.nparts, 
                      "dyn_Ms": None,
                      "lvm_Ms": None,
                      "optimize_joint": False,
                      "maxrun": 5,
                      "maxiter": 300,
                      "hold": None,}
    miter.params_range = [("hold", range(10)),
                          ("dyn_Ms", [(i,)*miter.nparts for i in [8]]),
                          ("lvm_Ms", [(i,)*miter.nparts for i in [10]]),]
    miter.directory = "./multiprimitive_{}".format(id)
    return miter


def analyze_stats():
    def firts_values(llst):
        return [lst[0] for lst in llst]

    for err_id in [0, 1]:
        stats = ErrorStatsReader()
        stats.errorsfilename = "/errors_mps({}).pkl".format(err_id)
        miter = create_model_iterator()
        miter.iterate_all_settings(stats.read_learned_errors)
        stats.params_range = miter.params_range
        print(stats.errs)
        
        # Analysis
        save_dir = "{}/statistics-mps_({})".format(miter.directory, err_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        statsfile = open(save_dir + "/stats-info.txt", "w") 
        # vCGPDM ELBO and MSE plots
        for key in ["ELBO", "MSE", "WRAP_DYN", "WRAP_PATH", "timing"]:
            #print(stats.errs)
            errs_axes = stats.params_range
            errs_value = stats.to_tensor(key=key, filter=None)
            #errs_axes, errs_value = select_by(errs_axes, errs_value, [("estimation_mode", EstimationMode.ELBO)])
            
            axes, data = errs_axes, np.squeeze(errs_value)
            means, std = mean_std(axes, data, alongs=["hold"])
            fig = plt.figure(figsize=(6, 5))
            plt.plot(0*data, data, "x", markersize=10)
            plt.errorbar(0*means, means, std, fmt='--o', capsize=2)
            plt.xlabel("dyn_Ms")
            plt.ylabel(key)
            plt.title(key)
            plot_dir = save_dir
            plt.savefig("{}/vCGPDM-{}.pdf".format(plot_dir, key))
            plt.close(fig)
            statsfile.write("Key: {}, means: {}, std: {}\n".format(key, means, std))
        statsfile.close()

                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    prog=__file__,
    description="""\
        Train multiprimitive vCGPDM crossvalidation models""",
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--v", action='version', version='%(prog)s 0.1')
    parser.add_argument("--i", type=int, default=None, help="""\
        run only i-th model. i in 1..IMAX. None to run all models. 0 to collect statistics""")
    args = parser.parse_args()
    i_model = args.i

    if i_model == None:
        miter = create_model_iterator()
        miter.iterate_all_settings(run_multiprimitive_CGPDM_crossvalidation)
    elif i_model == 0:
        analyze_stats()
    elif i_model <= 10:
        miter = create_model_iterator()
        miter.iterate_all_settings(run_multiprimitive_CGPDM_crossvalidation, i_model=i_model-1)
    