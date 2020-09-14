import os
import time
import numpy as np
from ml.gptheano.vecgpdm.enums import *
import ml.gptheano.vecgpdm.model as mdl
import numerical.numpytheano.theanopool as tp
import ml.gptheano.vecgpdm.kernels as krn
import numerical.numpyext.logger as npl
from validation.common import *
import validation.crossvalidate.common as vcc



class VCGPDMCrossval(vcc.ModelLerner):

    def __init__(self, dirname, param, args):
        super(VCGPDMCrossval, self).__init__(dirname, param, args)
        self.load_data()
        self.param = {"directory": None,
                      "mode": "ELBO",
                      "Qs": [3] * self.nparts,  # latent space dimensionality
                      "parts_IDs": self.parts_IDs,
                      "dyn": 5,
                      "lvm": 5,
                      "optimize_joint": False,
                      "maxrun": 4,
                      "maxiter": 300,
                      "bvh_nframes": 1000,  # number of frames to generate
                    }
        self.param.update(param)
        
        self.load_data()
        self.generate_BVHes()


    
    def create_model(self):
        # Configure loggers
        npl.setup_root_logger(rootlogfilename=os.path.join(self.dirname, "rootlog.pkl"))
        npl.setup_numpy_logger("ml.gptheano.vecgpdm.optimization",
                nplogfilename=os.path.join(self.dirname, "numpylog.pkl"))

        # Create variable pool        
        ns = tp.TheanoVarPool()

        # Create model data object
        self.training_y = [t.copy() for t in self.trial.training_data()]
        modeldata = mdl.ModelData(self.training_y, ns=ns)

        # Create model parameters object
        estimation_mode = EstimationMode.ELBO if self.param["mode"] == "ELBO" \
                else EstimationMode.MAP
  
        params = mdl.ModelParam(modeldata,
                                Qs=self.param["Qs"],
                                parts_IDs=self.param["parts_IDs"],
                                dyn_Ms=(self.param["dyn"],)*self.nparts,
                                lvm_Ms=(self.param["lvm"],)*self.nparts,
                                dyn_kern_type=krn.ARD_RBF_Kernel_noscale,
                                lvm_kern_type=krn.ARD_RBF_Kernel,
                                estimation_mode=estimation_mode,
                                ns=ns)

        # Create model
        self.model = mdl.VECGPDM(params, ns=ns)



    def train_model(self):
        self.model.precalc_posterior_predictive()
        mdl.save_all_plots(self.model, self.dirname, prefix="initial")
        
        if not self.args.dry:
            if self.param["optimize_joint"]:
                mdl.optimize_joint(self.model,
                                maxiter=self.param["maxiter"],
                                save_directory=self.dirname)
            else:
                mdl.optimize_blocked(self.model,
                                    maxrun=self.param["maxrun"],
                                    maxiter=self.param["maxiter"],
                                    print_vars=True,
                                    save_directory=self.dirname)
        


    def generate_BVHes(self):
        self.create_model()
        t0 = time.time()
        self.train_model()
        t1 = time.time()

        i = self.param["hold"]
        validation = self.trial.heldout_data()
        x0 = self.model.get_dynamics_start_point_by_training_chunk(
                training_chink_id=i, offset=0)

        T_validation = len(validation)
        x_generated = self.model.run_generative_dynamics(T_validation, x0)
        y_generated = self.model.lvm_map_to_observed(x_generated)
        errors = compute_errors(observed=validation,
                                predicted=np.hstack(y_generated))
        errors["ELBO"] = self.model.get_elbo_value()
        errors["timing"] = t1-t0
        errors["settings"] = self.param

        # Make a BVH
        if self.partitioner is not None:
            nframes = self.param["bvh_nframes"]
            x_generated = self.model.run_generative_dynamics(nframes, x0)
            y_generated = self.model.lvm_map_to_observed(x_generated)
            self.partitioner.set_all_parts_data(np.hstack(y_generated))
            self.partitioner.motiondata.write_BVH(
                    os.path.join(self.dirname, "final.bvh"))

        # Write the errors
        efn = os.path.join(self.dirname, "errors.pkl")
        with open(efn, "wb") as filehandle:
            pickle.dump(errors, filehandle)
