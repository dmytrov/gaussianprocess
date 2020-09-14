import os
import pickle
import numpy as np
import matplotlibex as plx
import ml.gptheano.vecgpdm.model as mdl
import ml.gptheano.vecgpdm.modelplots as mdlplt
import numerical.numpytheano.theanopool as tp
import numerical.numpytheano as nt
import matplotlibex.mlplot as plx
from ml.gptheano.vecgpdm.modelplots import MSEWriter



if __name__ == "__main__":
    df = open("coupled.pkl", "rb")
    pickled = pickle.load(df) 
    df.close()

    np.random.seed(100)

    X1 = pickled["latent 1"]
    X2 = pickled["latent 2"]
    Y1 = pickled["observed 1"]
    Y2 = pickled["observed 2"]

    Y_1_2 = np.hstack((Y1, Y2))
    #Y_1_2 = Y_1_2 + 0.1 * np.reshape(np.random.normal(size=Y_1_2.size), Y_1_2.shape)
    nsamples = 300
    Y = Y_1_2[:nsamples, :]
    
    ##############################################
    seed = 20
    np.random.seed(seed)
    y = Y
    parts_IDs = [0]*Y1.shape[1] + [1]*Y2.shape[1]
    nparts = np.max(parts_IDs) + 1
    dyn_M = 5  # number of dyn points
    lvm_M = 5  # number of lvm points
    Q = 2  # latent space dim
    dyn_Ms = [dyn_M] * nparts
    lvm_Ms = [lvm_M] * nparts
    Qs = [Q] * nparts
    
    directory = "synthetic/dyn({})-lvm({})-seed({})".format(dyn_M, lvm_M, seed)
    if not os.path.exists(os.path.join("./", directory)):
        os.makedirs(os.path.join("./", directory))

    plotter = MSEWriter()
    ns = tp.TheanoVarPool()
    data = mdl.ModelData(y, ns=ns)
    params = mdl.ModelParam(data, Qs=Qs, parts_IDs=parts_IDs, dyn_Ms=dyn_Ms, lvm_Ms=lvm_Ms, ns=ns)
    model = mdl.VECGPDM(params, ns=ns)
    
    plotter.save_dir = directory
    plotter(model)
    
    mdl.optimize_joint(model, 
            maxiter=1000, 
            save_directory=directory, 
            prefix="joint-first", 
            iter_callback=plotter)
    mdl.optimize_blocked(model,
            maxrun=10,
            maxiter=20,
            print_vars=True,
            save_directory=directory,
            iter_callback=plotter)
    mdl.optimize_joint(model, 
            maxiter=10000, 
            save_directory=directory, 
            prefix="joint-last", 
            iter_callback=plotter)
    
    

