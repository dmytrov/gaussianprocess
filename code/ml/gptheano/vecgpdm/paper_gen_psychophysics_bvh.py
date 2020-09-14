from __future__ import print_function
import sys
import os
import argparse
import multiprocessing as mps
import traceback
from shutil import copyfile
from time import strftime
import numpy as np
if "DISPLAY" not in os.environ:
    import matplotlib
    print("No DISPLAY found. Switching to noninteractive matplotlib backend...")
    print("Old backend is: {}".format(matplotlib.get_backend()))
    matplotlib.use('Agg')
    print("New backend is: {}".format(matplotlib.get_backend()))
import matplotlib.pyplot as plt
import matplotlibex.mlplot as plx
import numerical.numpytheano.theanopool as tp
import ml.dtw.dynamictimewarp as dtw
import ml.gptheano.vecgpdm.model as mdl
import ml.gptheano.vecgpdm.kernels as krn
import bvhrwroutines.bvhrwroutines as br
import pickle
from ml.gptheano.vecgpdm.enums import *
import numerical.numpyext.logger as npl
import dataset.mocap as dsm
import re



class TrainingParams(object):
    
    def __init__(self):
        self.naming_params = ["dataset", "dyn_M", "lvm_M"]
        self.root_dir = "."
        self.dry_run = False
        self.params = {"dataset": 0,
                "dyn_M": 2, 
                "lvm_M": 4, 
                "dyn_kern_boost":False, 
                "lvm_kern_boost":False,
                }


    def dir_name(self):
        return "-".join(["{}({})".format(param_name, self.params[param_name]) 
                for param_name in self.naming_params])

    
    def full_path(self, filename=""):
        return os.path.join(self.root_dir, self.dir_name(), filename)


    def from_dir_name(self, dir_name):
        pattern = "-".join(["{}\((?P<{}>\d*)\)".format(param_name, param_name) \
                for param_name in self.naming_params])
        m = re.match(pattern, dir_name)
        self.params.update(m.groupdict())
        

    @classmethod
    def test(cls):
        p = TrainingParams()
        assert p.dir_name() == "dataset(0)-dyn_M(2)-lvm_M(4)"
        p.params.update({"dyn_M": 50, "lvm_M": 6})
        assert p.dir_name() == "dataset(0)-dyn_M(50)-lvm_M(6)"
        p.from_dir_name("dataset(0)-dyn_M(20)-lvm_M(4)")
        assert p.dir_name() == "dataset(0)-dyn_M(20)-lvm_M(4)"
        
#TrainingParams.test()
#exit()

def load_motion(params, max_segments=all, save_training_bvh_dir=None):
    if params.params["dataset"] == 0:
        partitioner, parts_IDs, trials, starts_ends = dsm.load_recording(
            dsm.Recordings.exp3_walk, 
            bodypart_motiontypes=[(dsm.BodyParts.FullBody, dsm.MotionType.WALK_PHASE_ALIGNED)])
    elif params.params["dataset"] == 1:
        partitioner, parts_IDs, trials, starts_ends = dsm.load_recording(
            dsm.Recordings.exp3_walk, 
            bodypart_motiontypes=[(dsm.BodyParts.Upper, dsm.MotionType.WALK_PHASE_ALIGNED),
                                (dsm.BodyParts.Lower, dsm.MotionType.WALK_PHASE_ALIGNED)])
    y, parts_IDs = partitioner.get_all_parts_data_and_IDs()
    if max_segments == all:
        max_segments = len(starts_ends)
    ylist = dsm.select_sequences(partitioner, starts_ends[:max_segments])
    if save_training_bvh_dir is not None:
        for i, data in enumerate(ylist):
            filename = os.path.join(save_training_bvh_dir, 
                    "training({}).bvh".format(i))
            partitioner.set_all_parts_data(data)
            partitioner.motiondata.write_BVH(filename)
    return partitioner, parts_IDs, ylist



def write_bvh(partitioner, filename, data):
    partitioner.set_all_parts_data(data)
    partitioner.motiondata.write_BVH(filename)



def generate_bvh(partitioner, model, filename, nframes, startpoint):
    x_path = model.run_generative_dynamics(nframes, startpoint)
    y_path = model.lvm_map_to_observed(x_path)
    partitioner.set_all_parts_data(np.hstack(y_path))
    partitioner.motiondata.write_BVH(filename)



def compute_model_error(params, partitioner, model,
            startpoint_indexes, parts_IDs):
    std_err_sum = 0.0
    warp_err = 0.0
    n = 0
    pinds = br.IDs_to_indexes(parts_IDs)
    for i, i_pt in enumerate(startpoint_indexes):
        y_training = np.hstack([model.param.data.Y_sequences[i][:, pind] 
                for pind in pinds])
        nframes = y_training.shape[0]
        n += y_training.shape[0] * y_training.shape[1]
        startpoint = model.get_dynamics_start_point(i_pt)
        x_path = model.run_generative_dynamics(nframes * 2, startpoint)
        y_path = model.lvm_map_to_observed(x_path)
        y_path_array = np.hstack(y_path)
        odist, opath = dtw.dynTimeWarp(y_training, y_path_array)
        warp_err += odist
        wgen = np.array([y_path_array[i[0]] for i in opath])
        wtra = np.array([y_training[i[1]] for i in opath])
        err = np.sum((wgen - wtra)**2)
        print("Error: ", err)
        std_err_sum += err
    std_err = np.sqrt(std_err_sum / n)
    return std_err, warp_err



def model_trained(params):
    return os.path.exists(params.full_path("final_vars.pkl"))



def train_model(params, partitioner, model,
            startpoint_indexes, parts_IDs, nframes=1000):
    final_vars_filename = params.full_path("final_vars.pkl")
    if not model_trained(params):
        model_directory = params.full_path()
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model.precalc_posterior_predictive()
        mdl.save_all_plots(model, model_directory, prefix="initial")
        generate_bvh(partitioner, model, 
                filename=params.full_path("initial.bvh"), 
                nframes=nframes, startpoint=None)
        if not params.dry_run:
            mdl.optimize_blocked(model, maxrun=4, maxiter=300, print_vars=True, save_directory=model_directory)
        model.precalc_posterior_predictive()
        model.save_state_to(final_vars_filename)
        
        for i in startpoint_indexes:
            filename = params.full_path("final({}).bvh".format(i))
            startpoint = model.get_dynamics_start_point(i)
            generate_bvh(partitioner, model, filename=filename, nframes=nframes, startpoint=startpoint)

        

def save_model_stats(params, partitioner, model, startpoint_indexes, parts_IDs):
    std_err, warp_err = compute_model_error(params, partitioner, model,
            startpoint_indexes, parts_IDs)
    dyn_elbo = model.ns.evaluate(model.dyn_elbo)
    lvm_elbo = model.ns.evaluate(model.lvm_elbo)
    with open(params.full_path("final_stats.pkl"), "wb") as filehandle:
        pickle.dump({"std_err": std_err, "warp_err":warp_err, 
                "dyn_elbo":dyn_elbo, "lvm_elbo": lvm_elbo}, filehandle)
        

def create_model(params=None, max_segments=10):
    if params is None:
        params = TrainingParams()

    partitioner, parts_IDs, ylist = load_motion(params, max_segments)
     
    nparts = np.max(parts_IDs) + 1
    Qs = [3] * nparts
    dyn_M = params.params["dyn_M"]
    lvm_M = params.params["lvm_M"]
    dyn_Ms = [dyn_M] * nparts
    lvm_Ms = [lvm_M] * nparts
    startpoint_indexes = np.cumsum([0] + [chunk.shape[0] for chunk in ylist])[:-1]

    #ns = tp.NumpyVarPool()
    ns = tp.TheanoVarPool()
    data = mdl.ModelData(ylist, ns=ns)
    param = mdl.ModelParam(data, 
            Qs=Qs, 
            parts_IDs=parts_IDs, 
            dyn_Ms=dyn_Ms, 
            lvm_Ms=lvm_Ms,
            dyn_kern_type=krn.ARD_RBF_plus_linear_Kernel_noscale,
            lvm_kern_type=krn.ARD_RBF_plus_linear_Kernel,
            ns=ns)
    param.dyn_kern_boost = params.params["dyn_kern_boost"]
    param.lvm_kern_boost = params.params["lvm_kern_boost"]
    model = mdl.VECGPDM(param, ns=ns)
    model.param.set_Ms(dyn_Ms=[dyn_M] * nparts, lvm_Ms=[lvm_M] * nparts)
    return model, partitioner, parts_IDs, ylist, startpoint_indexes



def create_and_train_model(params, max_segments=10):
    if not os.path.exists(params.full_path()):
        os.makedirs(params.full_path())
    
    model, partitioner, parts_IDs, ylist, startpoint_indexes = \
        create_model(params, max_segments)    
    
    try:
        train_model(params, partitioner, model, startpoint_indexes, parts_IDs)
        save_model_stats(params, partitioner, model, 
                startpoint_indexes, parts_IDs)
        plt.close('all')    
    except (KeyboardInterrupt, SystemExit):
        raise
    except ArithmeticError as e:
        model.save_state_to(params.full_path("arithmetic_error.pkl"))
        pass
    except Exception as e:
        traceback.print_exc()
        raise


def collect_statistics(params=None, max_segments=10):    
    # plot dynamics time warp error
    print("+++ Computing errors from the stored models +++")
    dir_out = params.root_dir
    dir_names = [name for name in os.listdir(dir_out) if os.path.isdir(os.path.join(dir_out, name))]
    for dir_name in dir_names:
        tp = TrainingParams()
        tp.from_dir_name(dir_name)
        print(tp.params)
    exit()
    
    #model, partitioner, parts_IDs, ylist, startpoint_indexes = create_model(params, max_segments)
    #nparts = np.max(parts_IDs) + 1
    #initial_vars_state = model.ns.get_vars_state()

    
    std_err = np.zeros([dyn_range[-1], lvm_range[-1]])
    warp_err = np.zeros([dyn_range[-1], lvm_range[-1]])
    dyn_elbo = np.zeros([dyn_range[-1], lvm_range[-1]])
    lvm_elbo = np.zeros([dyn_range[-1], lvm_range[-1]])
    for dyn_M in dyn_range:
        for lvm_M in lvm_range:
            print("Reading model dyn({}) lvm({})".format(dyn_M, lvm_M))
            model.ns.set_vars_state(initial_vars_state)
            model.param.set_Ms(dyn_Ms=[dyn_M] * nparts, lvm_Ms=[lvm_M] * nparts)
            model_directory = "{}/dyn({})lvm({})".format(dir_out, dyn_M, lvm_M)
            model.load_state_from(model_directory + "/final_vars.pkl")
            model.precalc_posterior_predictive()
            se, we = compute_model_error(partitioner, model, dir_out, dyn_M, lvm_M, startpoint_indexes, parts_IDs)
            print("Standard error: {}, warp distance: {}".format(se, we))
            std_err[dyn_M-1, lvm_M-1] = se
            warp_err[dyn_M-1, lvm_M-1] = we
            dyn_elbo[dyn_M-1, lvm_M-1] = model.ns.evaluate(model.dyn_elbo)
            lvm_elbo[dyn_M-1, lvm_M-1] = model.ns.evaluate(model.lvm_elbo)
    std_err = std_err[dyn_range[0]-1:, lvm_range[0]-1:]
    warp_err = warp_err[dyn_range[0]-1:, lvm_range[0]-1:]
    dyn_elbo = dyn_elbo[dyn_range[0]-1:, lvm_range[0]-1:]
    lvm_elbo = lvm_elbo[dyn_range[0]-1:, lvm_range[0]-1:]

    with open(dir_out + "/std_err.pkl", "wb") as filehandle:
        pickle.dump(std_err, filehandle)
    with open(dir_out + "/warp_err.pkl", "wb") as filehandle:
        pickle.dump(warp_err, filehandle)
    with open(dir_out + "/dyn_elbo.pkl", "wb") as filehandle:
        pickle.dump(dyn_elbo, filehandle)
    with open(dir_out + "/lvm_elbo.pkl", "wb") as filehandle:
        pickle.dump(lvm_elbo, filehandle)

    plx.save_plot_matrix(dir_out, "walk_std_err",
                        std_err,
                        xticks=[lvm_range[0]-1, lvm_range[-1]],
                        yticks=[dyn_range[-1], dyn_range[0]-1],
                        xlabel="#z GPLVM",
                        ylabel="#z dynamics")
    plx.save_plot_matrix(dir_out, "walk_warp_err",
                        warp_err,
                        xticks=[lvm_range[0]-1, lvm_range[-1]],
                        yticks=[dyn_range[-1], dyn_range[0]-1],
                        xlabel="#z GPLVM",
                        ylabel="#z dynamics")
    plx.save_plot_matrix(dir_out, "walk_dyn_elbo",
                        dyn_elbo,
                        xticks=[lvm_range[0]-1, lvm_range[-1]],
                        yticks=[dyn_range[-1], dyn_range[0]-1],
                        xlabel="#z GPLVM",
                        ylabel="#z dynamics")
    plx.save_plot_matrix(dir_out, "walk_lvm_elbo",
                        lvm_elbo,
                        xticks=[lvm_range[0]-1, lvm_range[-1]],
                        yticks=[dyn_range[-1], dyn_range[0]-1],
                        xlabel="#z GPLVM",
                        ylabel="#z dynamics")
    plx.save_plot_matrix(dir_out, "walk_elbo",
                        lvm_elbo + dyn_elbo,
                        xticks=[lvm_range[0]-1, lvm_range[-1]],
                        yticks=[dyn_range[-1], dyn_range[0]-1],
                        xlabel="#z GPLVM",
                        ylabel="#z dynamics")
   
        

if __name__ == "__main__":   
    #npl.setup_root_logger()
    default_dir_out = "../../../../log/psychophysics" + \
            strftime("(%Y-%m-%d-%H.%M.%S)")
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="""\
            Genereates BVH files""",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--v", action='version', version='%(prog)s 0.1')
    parser.add_argument("--dir", type=str, default=default_dir_out, 
            help="output directory")
    parser.add_argument("--dataset", type=int, default=1, help="""\
        dataset ID. Default is 1""")
    parser.add_argument("--dyn", type=int, default=2, help="""\
        number of dynamics inducing points. Default is 2""")
    parser.add_argument("--lvm", type=int, default=4, help="""\
        number of latent variable model inducing points. Default is 4""")
    parser.add_argument("--stats", action="store_true", 
            help="collect the statistics from all the models")
    parser.add_argument("--t", action="store_true", 
            help="save training BVH files")
    parser.add_argument("--dry", action="store_true", 
            help="dry run, no optimiation")
    args = parser.parse_args()
    
    params = TrainingParams()
    params.dry_run = args.dry
    params.root_dir = args.dir
    params.params["dataset"] = args.dataset
    params.params["dyn_M"] = args.dyn
    params.params["lvm_M"] = args.lvm
        
    print("Output directory: {}".format(params.root_dir))

    if args.t:
        if not os.path.exists(args.dir):
            os.makedirs(args.dir)
        load_motion(params, save_training_bvh_dir=args.dir)
        exit()

    kern_boost = False
    if args.stats:
        # TODO:
        collect_statistics(params)
    else:
        if not model_trained(params):
            print("Training model: {}".format(params.dir_name()))
            create_and_train_model(params)
            print("==================================")
        else:
            print("Model {} is found, skipping...".format(params.dir_name()))
        

