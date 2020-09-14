import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlibex.mlplot as plx
import pickle



if __name__ == "__main__":
    directory = "../../../../generated/psychophysics(walk)"
    prefix = "iter_3"
    dyn_range = range(2, 17)
    lvm_range = range(4, 17)
    a = np.zeros([dyn_range[-1], lvm_range[-1]])
    for dyn_M in dyn_range:
        for lvm_M in lvm_range:
            model_directory = "{}/dyn({})lvm({})".format(directory, dyn_M, lvm_M)
            with open("{}/{}_alpha.txt".format(model_directory, prefix)) as f:
                f.readline()
                a[dyn_M-1, lvm_M-1] = float(f.readline())

    a = a[dyn_range[0]-1:, lvm_range[0]-1:]
    
    with open("./walk_elbo.pkl", "wb") as filehandle:
        pickle.dump(a, filehandle)

    plx.save_plot_matrix(".", "walk_ELBOs",
                         a, 
                         xticks=[lvm_range[0]-1, lvm_range[-1]],
                         yticks=[dyn_range[-1], dyn_range[0]-1],  
                         xlabel="#z GPLVM", 
                         ylabel="#z dynamics")
    