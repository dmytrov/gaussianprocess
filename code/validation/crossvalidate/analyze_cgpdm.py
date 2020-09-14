import os
import numpy as np
import matplotlib
if "DISPLAY" not in os.environ:
    print("No DISPLAY found. Switching to noninteractive matplotlib backend...")
    print("Old backend is: {}".format(matplotlib.get_backend()))
    matplotlib.use('Agg')
    print("New backend is: {}".format(matplotlib.get_backend()))
import matplotlib.pyplot as plt
from validation.common import *
from validation.crossvalidate_cgpdm import *


def firts_values(llst):
    return [lst[0] for lst in llst]


def good_settings(s):
    e = s["ELBO"]
    return e > -5.0e5 and e < 5.0e5 


if __name__ == "__main__":
    for dataset_id in (1, 2, 3, 4):
        stats = ErrorStatsReader()
        miter = create_model_iterator(dataset_id)

        # Analysis
        save_dir = miter.directory
        miter.iterate_all_settings(stats.read_learned_errors)
        stats.params_range = miter.params_range

        # vCGPDM ELBO and MSE plots
        for key in ["MSE", "WRAP_DYN", "WRAP_PATH", "timing"]:
            errs_axes = stats.params_range
            errs_value = stats.to_tensor(key=key)
            axes, data = select_by(errs_axes, errs_value,
                [("estimation_mode", EstimationMode.MAP),
                (stats.params_range[1][0], stats.params_range[1][1][1]), # ("dyn_Ms", (4,)),
                (stats.params_range[2][0], stats.params_range[2][1][1])])  # ("lvm_Ms", (4,)),
            means, std = mean_std(axes, data, alongs=["hold"])
            fig = plt.figure(figsize=(6, 5))
            plt.plot(0 * data, data, "x", markersize=10)
            plt.errorbar(0, means, std, fmt='--o', capsize=2)
            plt.ylabel(key)
            plt.title("MAP CGPDM, {}".format(key))
            plot_dir = "{}/statistics".format(save_dir)
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig("{}/MAP-CGPDM-{}.pdf".format(plot_dir, key))
            #plt.show()
            plt.close(fig)

        # vCGPDM ELBO and MSE plots
        for key in ["ELBO", "MSE", "WRAP_DYN", "WRAP_PATH", "timing"]:
            errs_axes = stats.params_range
            errs_value = stats.to_tensor(key=key, filter=good_settings)
            errs_axes, errs_value = select_by(errs_axes, errs_value, [("estimation_mode", EstimationMode.ELBO)])
            for iter_comb, axes, data in iterate_by(errs_axes, errs_value, iter_params_keys=["lvm_Ms"]):
                means, std = mean_std(axes, data, alongs=["hold"])
                fig = plt.figure(figsize=(6, 5))
                plt.plot(firts_values(values_by_name(axes, "dyn_Ms")), data, "x", markersize=10)
                plt.errorbar(firts_values(values_by_name(axes, "dyn_Ms")), means, std, fmt='--o', capsize=2)
                plt.xlabel("dyn_Ms")
                plt.ylabel(key)
                plt.title("vCGPDM, {} \n Model parameters: {}".format(key, iter_comb))
                plot_dir = "{}/statistics".format(save_dir)
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                plt.savefig("{}/vCGPDM-{}-parameters({}).pdf".format(plot_dir, key, iter_comb))
                #plt.show()
                plt.close(fig)




