import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ml.gptheano.vecgpdm.model as mdl
import numerical.numpytheano.theanopool as tp
import ml.gptheano.vecgpdm.kernels as krn
import numerical.numpytheano as nt
from ml.gptheano.vecgpdm.enums import *



if __name__ == "__main__":
    t = np.linspace(0.0, 3.1*2*np.pi, num=100)
    f1 = 1.0
    f2 = 1.0
    f3 = 3.14
    y1 = np.vstack((5.0*np.sin(f1*t+0.0), 5.0*np.sin(f1*t+1.5),
                    4.1*np.sin(f1*t+0.4), 4.1*np.sin(f1*t+1.8),
                    3.1*np.sin(f1*t+0.4), 3.1*np.sin(f1*t+1.8),
                    2.1*np.sin(f1*t+0.4), 2.1*np.sin(f1*t+1.8),

                    5.0*np.sin(f2*t+0.0), 5.0*np.sin(f2*t+3.5),
                    4.1*np.sin(f2*t+0.4), 4.1*np.sin(f2*t+2.8),
                    3.1*np.sin(f2*t+0.4), 3.1*np.sin(f2*t+3.8),
                    2.1*np.sin(f2*t+0.4), 2.1*np.sin(f2*t+4.8),
      
                    5.0*np.sin(f3*t+0.0), 5.0*np.sin(f3*t+3.5),
                    4.1*np.sin(f3*t+0.4), 4.1*np.sin(f3*t+2.8),
                    3.1*np.sin(f3*t+0.4), 3.1*np.sin(f3*t+3.8),
                    2.1*np.sin(f3*t+0.4), 2.1*np.sin(f3*t+4.8)
                    )).T


    np.random.seed(0)
    # Add noise and displacement
    y = [y1 + 0.2*np.reshape(np.random.normal(size=y1.size), y1.shape) \
            + 10.0 * np.random.normal(size=y1.shape[1])]
    parts_IDs = [0]*8 + [1]*8 + [2]*8
    nparts = np.max(parts_IDs) + 1
    dyn_Ms = [8] * nparts
    lvm_Ms = [8] * nparts
    Qs = [2] * nparts
    estimation_mode=EstimationMode.MAP
                            
    
    directory = "sines"
    
    print("|=========== NumPy ==========|")
    ns = tp.NumpyVarPool()
    data = mdl.ModelData(y, ns=ns)
    params = mdl.ModelParam(data, 
                            Qs=Qs, 
                            parts_IDs=parts_IDs, 
                            dyn_Ms=dyn_Ms, 
                            lvm_Ms=lvm_Ms, 
                            lvm_kern_type=krn.RBF_Kernel, 
                            estimation_mode=estimation_mode,
                            ns=ns)
    model = mdl.VECGPDM(params, ns=ns)
    model.precalc_posterior_predictive()
    
    #mdl.plot_latent_space(model)
    mdl.save_plot_latent_space(model, directory, prefix="initial")
    
    print("|=========== Theano ==========|")
    ns = tp.TheanoVarPool()
    data = mdl.ModelData(y, ns=ns)
    params = mdl.ModelParam(data, 
                            Qs=Qs, 
                            parts_IDs=parts_IDs, 
                            dyn_Ms=dyn_Ms, 
                            lvm_Ms=lvm_Ms, 
                            lvm_kern_type=krn.RBF_Kernel, 
                            estimation_mode=estimation_mode,
                            ns=ns)
    model = mdl.VECGPDM(params, ns=ns)
    model.precalc_posterior_predictive()
    
    mdl.save_plot_latent_space(model, directory, prefix="initial")
    mdl.save_plot_latent_vs_generated(model, directory, prefix="initial")
    mdl.save_plot_training_vs_generated(model, directory, prefix="initial")

    mdl.optimize_blocked(model, niterations=3, maxiter=30, print_vars=True, save_directory=directory)
    


