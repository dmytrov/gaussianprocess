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
from validation.crossvalidate_dynamic_mp import *





if __name__ == "__main__":
    for dataset_id in (0, 1, 2):
        stats = ErrorStatsReader()
        miter = create_model_iterator(dataset_id)

        # Analysis
        save_dir = miter.directory
        miter.iterate_all_settings(stats.read_learned_errors)
        stats.params_range = miter.params_range

        for key in ["MSE", "WRAP_DYN", "WRAP_PATH", "timing"]:
            errs_axes = stats.params_range
            errs_value = stats.to_tensor(key=key)
            axes, data = errs_axes, errs_value
            means, std = mean_std(axes, data, alongs=["hold"])
            fig = plt.figure(figsize=(6, 5))
            plt.plot(values_by_name(axes, "npsi"), data, "x", markersize=10)
            plt.errorbar(values_by_name(axes, "npsi"), means, std, fmt='--o', capsize=2)
            plt.xlabel("npsi")
            plt.ylabel(key)
            plt.title("DMP, {}".format(key))
            plot_dir = "{}/statistics".format(save_dir)
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig("{}/DMP-{}.pdf".format(plot_dir, key))
            #plt.show()
            plt.close(fig)

