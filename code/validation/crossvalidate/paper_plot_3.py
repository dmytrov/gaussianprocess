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
import validation.crossvalidate_cgpdm as cgpdm
import validation.crossvalidate_temporal_mp as tmp


def firts_values(llst):
    return [lst[0] for lst in llst]


def good_settings(s):
    elbo = s["ELBO"]
    mse = s["MSE"]
    return elbo > -5.0e5 and elbo < 3.0e4 and mse < 0.015


if __name__ == "__main__":
    fig = plt.figure(figsize=(5, 5))
    save_dir = "."
        
    # vCGPDM
    for dataset_id in [4]:
        stats = ErrorStatsReader()
        miter = cgpdm.create_model_iterator(dataset_id)

        miter.iterate_all_settings(stats.read_learned_errors)
        stats.params_range = miter.params_range

        # MSE vs ELBO plot
        MSE_axes, MSE_value = select_by(
            stats.params_range,
            stats.to_tensor(key="MSE", filter=good_settings),
            [("estimation_mode", cgpdm.EstimationMode.ELBO)])
        MSE_means, MSE_std = mean_std(MSE_axes, MSE_value, alongs=["hold"])

        ELBO_axes, ELBO_value = select_by(
            stats.params_range,
            stats.to_tensor(key="ELBO", filter=good_settings),
            [("estimation_mode", cgpdm.EstimationMode.ELBO)])
        ELBO_means, ELBO_std = mean_std(ELBO_axes, ELBO_value, alongs=["hold"])
        #plt.plot(ELBO_means, MSE_means, "x", color="black", alpha=0.5)

        print("MSE_std", MSE_std)
        print("ELBO_std", ELBO_std)
        fmts = ("o", "^", "s", "D", "x", "*")
        for lvm_index in range(ELBO_means.shape[1]):
            #plt.plot(ELBO_means[:, lvm_index], MSE_means[:, lvm_index], "-x",
            #         label="#lvm_IPs={}".format(ELBO_axes[1][1][lvm_index][0]))
            plt.errorbar(ELBO_means[:, lvm_index], MSE_means[:, lvm_index], fmt=fmts[lvm_index],
                         xerr=ELBO_std[:, lvm_index], yerr=MSE_std[:, lvm_index],
                         elinewidth=0.5, alpha=0.9,
                         label="#lvm IPs={}".format(ELBO_axes[1][1][lvm_index][0]))

        
    # Temporal MPs
    for dataset_id in []:
        stats = ErrorStatsReader()
        miter = tmp.create_model_iterator(dataset_id)
        miter.iterate_all_settings(stats.read_learned_errors)
        stats.params_range = miter.params_range

        # MSE vs ELBO plot
        MSE_means, MSE_std = mean_std(stats.params_range, stats.to_tensor(key="MSE"), alongs=["hold"])
        ELBO_means, ELBO_std = mean_std(stats.params_range, stats.to_tensor(key="ELBO"), alongs=["hold"])
        plt.plot(ELBO_means, MSE_means, "o", color="black", alpha=0.5)

    #plt.legend(bbox_to_anchor=(1.1, 1.0))
    plt.legend()
    plt.title("vCGPDM MSE vs ELBO, 'Walk+wave', U+L")
    plt.xlabel("ELBO")
    plt.ylabel("MSE")

    filename = "vcgpdm_mse_vs_elbo_errorbars_noline.pdf"
    fig.subplots_adjust(left=0.15)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close(fig)

