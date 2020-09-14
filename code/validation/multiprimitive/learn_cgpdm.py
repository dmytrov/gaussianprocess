import os
import sys
import time
import logging
import pickle
import numpy as np
import matplotlib
if "DISPLAY" not in os.environ:
    print("No DISPLAY found. Switching to noninteractive matplotlib backend...")
    print("Old backend is: {}".format(matplotlib.get_backend()))
    matplotlib.use('Agg')
    print("New backend is: {}".format(matplotlib.get_backend()))
import matplotlib.pyplot as plt
import ml.gptheano.vecgpdm.model as mdl
import numerical.numpytheano.theanopool as tp
import ml.gptheano.vecgpdm.kernels as krn
import numerical.numpyext.logger as npl
from ml.gptheano.vecgpdm.enums import *
import dataset.mocap as ds
from validation.common import *
import argparse


def train_CGPDM(y, settings):
    directory = settings["directory"]
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    statefilename = "{}/CGPDM_learned.pkl".format(directory)
    if not os.path.exists(statefilename):
        npl.setup_root_logger(rootlogfilename="{}/rootlog.txt".format(directory))
        npl.setup_numpy_logger("ml.gptheano.vecgpdm.optimization",
            nplogfilename="{}/numpylog.pkl".format(directory))

    ns = tp.TheanoVarPool()
    data = mdl.ModelData(y, ns=ns)
    params = mdl.ModelParam(data,
                            Qs=settings["Qs"],
                            parts_IDs=settings["parts_IDs"],
                            dyn_Ms=settings["dyn_Ms"],
                            lvm_Ms=settings["lvm_Ms"],
                            lvm_kern_type=krn.RBF_Kernel,
                            estimation_mode=settings["estimation_mode"],
                            ns=ns)
    model = mdl.VECGPDM(params, ns=ns)

    if not os.path.exists(statefilename):
        model.precalc_posterior_predictive()
        mdl.save_plot_latent_space(model, directory, prefix="initial")
        mdl.save_plot_latent_vs_generated(
            model, directory, prefix="initial")
        mdl.save_plot_training_vs_generated(
            model, directory, prefix="initial")
        if not settings["dry_run"]:
            if settings["optimize_joint"]:
                mdl.optimize_joint(model,
                                maxiter=settings["maxiter"],
                                save_directory=directory)
            else:
                mdl.optimize_blocked(model,
                                    maxrun=settings["maxrun"],
                                    maxiter=settings["maxiter"],
                                    print_vars=True,
                                    save_directory=directory)
        model.save_state_to(statefilename)
    else:
        model.load_state_from(statefilename)

    model.precalc_posterior_predictive()
    return model


def run_CGPDM_crossvalidation(training, validation, settings, bvhpartitioner=None):
    errorsfilename = settings["directory"] + "/errors.pkl"
    if not os.path.exists(errorsfilename):
        if not os.path.exists(settings["directory"]):
            os.makedirs(settings["directory"])

        y = [t.copy() for t in training]
        # Validation seed is at the end
        y.append(validation[:settings["validation_seed_size"], :])

        t0 = time.time()
        model = train_CGPDM(y, settings=settings)
        t1 = time.time()

        dyn_order = 2
        validation_skip = settings["validation_seed_size"] - dyn_order
        T_validation = validation.shape[0] - validation_skip
        datasize = model.param.data.N
        x0 = model.get_dynamics_start_point(datasize - dyn_order)  # Held-out data primer is at the end
        x_generated = model.run_generative_dynamics(T_validation, x0)
        y_generated = model.lvm_map_to_observed(x_generated)
        errors = compute_errors(observed=validation[validation_skip:, :],
                                predicted=np.hstack(y_generated))
        errors["ELBO"] = model.get_elbo_value()
        errors["timing"] = t1-t0
        errors["settings"] = settings

        # Make a BVH
        if bvhpartitioner is not None:
            nframes = settings["bvh_nframes"]
            x_generated = model.run_generative_dynamics(nframes, x0)
            y_generated = model.lvm_map_to_observed(x_generated)
            bvhpartitioner.set_all_parts_data(np.hstack(y_generated))
            bvhpartitioner.motiondata.write_BVH(settings["directory"] + "/final.bvh")

        # Write the errors
        with open(errorsfilename, "wb") as filehandle:
            pickle.dump(errors, filehandle)


def print_settings(training, validation, settings, bvhpartitioner=None):
    print(settings["directory"])


def create_model_iterator(id=1):
    if id == 1:
        miter = ModelIterator()
        miter.load_recording(
            ds.Recordings.exp3_walk,
            bodypart_motiontypes=[(ds.BodyParts.Upper, ds.MotionType.WALK_PHASE_ALIGNED),
                                  (ds.BodyParts.Lower, ds.MotionType.WALK_PHASE_ALIGNED)],
            max_chunks=10)
    elif id == 2:
        miter = ModelIterator()
        miter.load_recording(
            ds.Recordings.exp3_protest_walk_2arms,
            bodypart_motiontypes=[(ds.BodyParts.Upper, ds.MotionType.WALK_SHAKE_ARMS),
                                  (ds.BodyParts.Lower, ds.MotionType.WALK_SHAKE_ARMS)],
            max_chunks=10)
        
    miter.settings = {"recording_info": miter.recording.info,
                      "recording_filename": miter.recording.filename,
                      "directory": None,
                      "estimation_mode": EstimationMode.ELBO,
                      "validation_seed_size": 4,  # number of frames to be included into the training set
                      "Qs": [3] * miter.nparts,  # latent space dimensionality
                      "parts_IDs": miter.parts_IDs,
                      "dyn_Ms": None,
                      "lvm_Ms": None,
                      "optimize_joint": False,
                      "maxrun": 3,
                      "maxiter": 300,
                      "bvh_nframes": 300,  # number of frames to generate
                      "dry_run": False,  # skip optimization completely
                      "hold": None,}
    miter.params_range = [("estimation_mode", [EstimationMode.ELBO]),
                          ("dyn_Ms", [(i,)*miter.nparts for i in [8]]),
                          ("lvm_Ms", [(i,)*miter.nparts for i in [10]]),
                          ("hold", range(miter.trial.nchunks()))]
    miter.directory = "./cgpdm_{}".format(id)
    return miter


def analyze_stats():
    def firts_values(llst):
        return [lst[0] for lst in llst]

    for dataset_id in (1, 2):
        stats = ErrorStatsReader()
        miter = create_model_iterator(dataset_id)

        # Analysis
        save_dir = miter.directory
        miter.iterate_all_settings(stats.read_learned_errors)
        stats.params_range = miter.params_range

        plot_dir = "{}/statistics".format(save_dir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
                
        statsfile = open(plot_dir + "/stats-info.txt", "w") 
        # vCGPDM ELBO and MSE plots
        for key in ["ELBO", "MSE", "WRAP_DYN", "WRAP_PATH", "timing"]:
            errs_axes = stats.params_range
            errs_value = stats.to_tensor(key=key, filter=None)
            errs_axes, errs_value = select_by(errs_axes, errs_value, [("estimation_mode", EstimationMode.ELBO)])
            for iter_comb, axes, data in iterate_by(errs_axes, errs_value, iter_params_keys=["lvm_Ms"]):
                means, std = mean_std(axes, data, alongs=["hold"])
                fig = plt.figure(figsize=(6, 5))
                plt.plot(firts_values(values_by_name(axes, "dyn_Ms")), data, "x", markersize=10)
                plt.errorbar(firts_values(values_by_name(axes, "dyn_Ms")), means, std, fmt='--o', capsize=2)
                plt.xlabel("dyn_Ms")
                plt.ylabel(key)
                plt.title("vCGPDM, {} \n Model parameters: {}".format(key, iter_comb))
                plt.savefig("{}/vCGPDM-{}-parameters({}).pdf".format(plot_dir, key, iter_comb))
                plt.close(fig)

                statsfile.write("Key: {}, means: {}, std: {}\n".format(key, means, std))
        statsfile.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    prog=__file__,
    description="""\
        Train vCGPDM crossvalidation models""",
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--v", action='version', version='%(prog)s 0.1')
    parser.add_argument("--i", type=int, default=None, help="""\
        run only i-th model. i in 1..IMAX. None to run all models. 0 to collect statistics""")
    args = parser.parse_args()
    i_model = args.i

    if i_model == None:
        for dataset_id in [1, 2]:
            miter = create_model_iterator(dataset_id)
            miter.iterate_all_settings(run_CGPDM_crossvalidation)
    elif i_model == 0:
        analyze_stats()
    elif i_model <= 10:
        miter = create_model_iterator(1)
        miter.iterate_all_settings(run_CGPDM_crossvalidation, i_model=i_model-1)
    elif i_model <= 20:
        miter = create_model_iterator(2)
        miter.iterate_all_settings(run_CGPDM_crossvalidation, i_model=i_model-10-1)
    


