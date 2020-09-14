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
import validation.crossvalidate.crossvalidate_cgpdm as cgpdm
import validation.crossvalidate.crossvalidate_temporal_mp as tmp
import validation.crossvalidate.crossvalidate_dynamic_mp as dmp


def firts_values(llst):
    return [lst[0] for lst in llst]


def good_settings_1(s):
    elbo = s["ELBO"]
    mse = s["MSE"]
    return elbo > -5.0e5 and elbo < 3.0e4 and mse < 0.2

def good_settings_2(s):
    elbo = s["ELBO"]
    mse = s["MSE"]
    return elbo > -5.0e5 and elbo < 3.0e4 and mse < 0.08


def process_datasets(
        vcgpd_ds_ids, cgpd_ds_ids=None,
        tmp_ds_ids=None, dmp_ds_ids=None,
        plot_best_elbo=False,
        plot_tmp_elbo=True,
        settings_filter=None,
        title=None, filename=None):
    fig, ax1 = plt.subplots(figsize=(5, 5))
    #fig = plt.figure(figsize=(7, 7))
    #save_dir = "."

    model_dataset_names = []
    best_elbo_infos = []
    best_elbo_means = []
    best_elbo_stds = []
    best_elbo_mse_means = []
    best_elbo_mse_stds = []
    best_mse_infos = []
    best_mse_means = []
    best_mse_stds = []
    best_mse_elbo_means = []
    best_mse_elbo_stds = []
    training_time_mean = []
    training_time_std = []

    mse_key = "MSE"
    #mse_key = "WRAP_PATH"

    # vCGPDM
    datasets = ("NONE", "vGPDM", "vCGPDM,\nU+L", "vGPDM", "vCGPDM,\nU+L")
    for dataset_id in (vcgpd_ds_ids):
        stats = ErrorStatsReader()
        miter = cgpdm.create_model_iterator(dataset_id)

        miter.iterate_all_settings(stats.read_learned_errors)
        stats.params_range = miter.params_range

        # MSE vs ELBO plot
        MSE_axes, MSE_value = select_by(
            stats.params_range,
            stats.to_tensor(key=mse_key, filter=settings_filter),
            [("estimation_mode", cgpdm.EstimationMode.ELBO)])
        MSE_means, MSE_stds = mean_std(MSE_axes, MSE_value, alongs=["hold"])

        ELBO_axes, ELBO_value = select_by(
            stats.params_range,
            stats.to_tensor(key="ELBO", filter=settings_filter),
            [("estimation_mode", cgpdm.EstimationMode.ELBO)])
        ELBO_means, ELBO_stds = mean_std(ELBO_axes, ELBO_value, alongs=["hold"])
        
        # Dataset info
        model_dataset_names.append(datasets[dataset_id])

        # Best ELBO
        i, j = np.unravel_index(np.nanargmax(ELBO_means), ELBO_means.shape)
        best_elbo_infos.append("argmax(ELBO)=({}={},{}={})".format(
            ELBO_axes[0][0],
            firts_values(ELBO_axes[0][1])[i],
            ELBO_axes[1][0],
            firts_values(ELBO_axes[1][1])[j]))
        best_elbo_means.append(ELBO_means[i, j])
        best_elbo_stds.append(ELBO_stds[i, j])
        best_elbo_mse_means.append(MSE_means[i, j])
        best_elbo_mse_stds.append(MSE_stds[i, j])

        # Best MSE
        i, j = np.unravel_index(np.nanargmin(MSE_means), ELBO_means.shape)
        best_mse_infos.append("argmin(MSE)=({}={},{}={})".format(
            ELBO_axes[0][0],
            firts_values(ELBO_axes[0][1])[i],
            ELBO_axes[1][0],
            firts_values(ELBO_axes[1][1])[j]))
        best_mse_means.append(MSE_means[i, j])
        best_mse_stds.append(MSE_stds[i, j])
        best_mse_elbo_means.append(ELBO_means[i, j])
        best_mse_elbo_stds.append(ELBO_stds[i, j])

        # Timing
        timing_axes, timing_value = select_by(
            stats.params_range,
            stats.to_tensor(key="timing", filter=settings_filter),
            [("estimation_mode", cgpdm.EstimationMode.ELBO)])
        print(timing_axes)
        
        timing_mean, timing_stds = mean_std(timing_axes, timing_value, alongs=["hold", "dyn_Ms", "lvm_Ms"])
        training_time_mean.append(timing_mean)
        training_time_std.append(timing_stds)
        

    # MAP CGPDM
    datasets = ("NONE", "MAP GPDM", "MAP CGPDM,\nU+L", "MAP GPDM", " MAP CGPDM,\nU+L")
    for dataset_id in (vcgpd_ds_ids):
        stats = ErrorStatsReader()
        miter = cgpdm.create_model_iterator(dataset_id)

        miter.iterate_all_settings(stats.read_learned_errors)
        stats.params_range = miter.params_range

        # MSE vs ELBO plot
        errs_axes = stats.params_range
        errs_value = stats.to_tensor(key=mse_key)
        axes, data = select_by(errs_axes, errs_value,
                [("estimation_mode", cgpdm.EstimationMode.MAP),
                (stats.params_range[1][0], stats.params_range[1][1][1]), # ("dyn_Ms", (4,)),
                (stats.params_range[2][0], stats.params_range[2][1][1])])  # ("lvm_Ms", (4,)),
        means, std = mean_std(axes, data, alongs=["hold"])

        # Dataset info
        model_dataset_names.append(datasets[dataset_id])

        # Best ELBO
        best_elbo_infos.append(None)
        best_elbo_means.append(0)
        best_elbo_stds.append(0)
        best_elbo_mse_means.append(0)
        best_elbo_mse_stds.append(0)

        # Best MSE
        best_mse_infos.append("")
        best_mse_means.append(means)
        best_mse_stds.append(std)
        best_mse_elbo_means.append(0)
        best_mse_elbo_stds.append(0)

        # Timing
        errs_value = stats.to_tensor(key="timing")
        timing_axes, timing_data = select_by(errs_axes, errs_value,
                [("estimation_mode", cgpdm.EstimationMode.MAP),
                (stats.params_range[1][0], stats.params_range[1][1][1]), # ("dyn_Ms", (4,)),
                (stats.params_range[2][0], stats.params_range[2][1][1])])  # ("lvm_Ms", (4,)),
        print(timing_axes)
        print(timing_data)
        
        timing_mean, timing_stds = mean_std(timing_axes, timing_data, alongs=["hold"])
        training_time_mean.append(timing_mean)
        training_time_std.append(timing_stds)

        print(training_time_mean)
        #exit()

    # Temporal MPs
    datasets = ("NONE", "TMP", "TMP")
    for dataset_id in (tmp_ds_ids):
        stats = ErrorStatsReader()
        miter = tmp.create_model_iterator(dataset_id)
        miter.iterate_all_settings(stats.read_learned_errors)
        stats.params_range = miter.params_range

        # MSE vs ELBO plot
        MSE_axes, MSE_value = stats.params_range, stats.to_tensor(key=mse_key)
        MSE_means, MSE_stds = mean_std(MSE_axes, MSE_value, alongs=["hold"])
        ELBO_axes, ELBO_value = stats.params_range, stats.to_tensor(key="ELBO")
        ELBO_means, ELBO_stds = mean_std(ELBO_axes, ELBO_value, alongs=["hold"])

        # Dataset info
        model_dataset_names.append(datasets[dataset_id])

        if plot_tmp_elbo:
            # Best ELBO
            i = np.nanargmax(ELBO_means)
            best_elbo_infos.append("argmax(ELBO)=({}={})".format(ELBO_axes[0][0], ELBO_axes[0][1][i]))
            best_elbo_means.append(ELBO_means[i])
            best_elbo_stds.append(ELBO_stds[i])
            best_elbo_mse_means.append(MSE_means[i])
            best_elbo_mse_stds.append(MSE_stds[i])
        
        else:
            best_elbo_infos.append(None)
            best_elbo_means.append(0)
            best_elbo_stds.append(0)
            best_elbo_mse_means.append(0)
            best_elbo_mse_stds.append(0)

        # Best MSE
        i = np.nanargmin(MSE_means)
        best_mse_infos.append("argmin(MSE)=({}={})".format(ELBO_axes[0][0], ELBO_axes[0][1][i]))
        best_mse_means.append(MSE_means[i])
        best_mse_stds.append(MSE_stds[i])

        if plot_tmp_elbo:
            best_mse_elbo_means.append(ELBO_means[i])
            best_mse_elbo_stds.append(ELBO_stds[i])
        else:
            best_mse_elbo_means.append(0)
            best_mse_elbo_stds.append(0)

        

        # Timing
        timing_axes, timing_value = stats.params_range, stats.to_tensor(key="timing")
        print(timing_axes)
        
        timing_mean, timing_stds = mean_std(timing_axes, timing_value, alongs=["hold", "numprim"])
        training_time_mean.append(timing_mean)
        training_time_std.append(timing_stds)

    # Dynamic MPs
    datasets = ("NONE", "DMP", "DMP")
    for dataset_id in (dmp_ds_ids):
        stats = ErrorStatsReader()
        miter = dmp.create_model_iterator(dataset_id)
        miter.iterate_all_settings(stats.read_learned_errors)
        stats.params_range = miter.params_range

        MSE_axes, MSE_value = stats.params_range, stats.to_tensor(key=mse_key)
        MSE_means, MSE_stds = mean_std(MSE_axes, MSE_value, alongs=["hold"])

        # Dataset info
        model_dataset_names.append(datasets[dataset_id])

        # Best ELBO
        best_elbo_infos.append(None)
        best_elbo_means.append(0)
        best_elbo_stds.append(0)
        best_elbo_mse_means.append(0)
        best_elbo_mse_stds.append(0)

        # Best MSE
        i = np.nanargmin(MSE_means)
        best_mse_infos.append("argmin(MSE)=({}={})".format(MSE_axes[0][0], MSE_axes[0][1][i]))
        best_mse_means.append(MSE_means[i])
        best_mse_stds.append(MSE_stds[i])
        best_mse_elbo_means.append(0)
        best_mse_elbo_stds.append(0)

        # Timing
        timing_axes, timing_value = stats.params_range, stats.to_tensor(key="timing")
        print(timing_axes)
        
        timing_mean, timing_stds = mean_std(timing_axes, timing_value, alongs=["hold", "npsi"])
        training_time_mean.append(timing_mean)
        training_time_std.append(timing_stds)



    print(model_dataset_names)
    print(best_elbo_infos)
    print(best_elbo_means)
    print(best_elbo_stds)
    print(best_mse_infos)
    print(best_mse_means)
    print(best_mse_stds)

    ind = np.arange(len(model_dataset_names))
    width = 0.2
    ax2 = ax1.twinx()

    if plot_best_elbo:
        rects1 = ax1.bar(ind+1*width, best_mse_means, width,
                        color='fuchsia', edgecolor='black', linewidth=1,
                        hatch="*",
                        yerr=best_mse_stds,
                        error_kw=dict(elinewidth=2, ecolor='black'))
        rects2 = ax1.bar(ind+2*width, best_elbo_mse_means, width,
                        color='lightgreen', edgecolor='black', linewidth=1,
                        hatch=".",
                        yerr=best_elbo_mse_stds,
                        error_kw=dict(elinewidth=2, ecolor='black'))
        rects3 = ax2.bar(ind+3.1*width, best_elbo_means, width,
                        color='dodgerblue', edgecolor='black', linewidth=1,
                        hatch="x",
                        yerr=best_elbo_stds,
                        error_kw=dict(elinewidth=2, ecolor='black'))
        rects4 = ax2.bar(ind+4.1*width, best_mse_elbo_means, width,
                        color='greenyellow', edgecolor='black', linewidth=1,
                        hatch="o",
                        yerr=best_mse_elbo_stds,
                        error_kw=dict(elinewidth=2, ecolor='black'))
        
    else:
        rects1 = ax1.bar(ind+width, best_mse_means, width,
                        color='hotpink', ecolor='black',
                        yerr=best_mse_stds,
                        error_kw=dict(elinewidth=2, ecolor='black'))
        rects2 = ax2.bar(ind+2*width, best_mse_elbo_means, width,
                        color='deepskyblue', ecolor='black', hatch="//",
                        yerr=best_mse_elbo_stds,
                        error_kw=dict(elinewidth=2, ecolor='black'))
    
    ax1.set_ylabel("MSE")
    ax2.set_ylabel("ELBO")
    ax1.set_ylim(bottom=0)

    xTickMarks = model_dataset_names
    ax1.set_xticks(ind+width)
    xtickNames = ax1.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=90, fontsize=10)
    if plot_best_elbo:
        ax2.legend((rects1[0], rects2[0], rects3[0], rects4[0]), 
            ("Best MSE", "Best ELBO's MSE", "Best ELBO", "Best MSE's ELBO"))
    else:
        ax2.legend((rects1[0], rects2[0]), ("Best MSE", "Best MSE's ELBO"))
    ax2.get_yaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    if title is not None:
        plt.title(title)

    fig.subplots_adjust(bottom=0.26, left=0.15, right=0.85)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename + ".pdf")
        plt.close(fig)

    # Timing
    print(model_dataset_names,
            training_time_mean,  
            training_time_std)

    fig, ax1 = plt.subplots(figsize=(5, 5))
    rects1 = ax1.bar(
            range(len(model_dataset_names)),
            training_time_mean,  
            yerr=training_time_std,
            edgecolor='black', linewidth=1,)
    ax1.set_ylabel("Time, s")
    ax1.set_xticks(np.arange(len(model_dataset_names)))
    xtickNames = ax1.set_xticklabels(model_dataset_names)
    plt.setp(xtickNames, rotation=90, fontsize=10)
    plt.title("Training time")
    fig.subplots_adjust(bottom=0.26, left=0.15, right=0.85)
    plt.savefig(filename + "-timing.pdf")
    plt.close(fig)


if __name__ == "__main__":
    process_datasets(vcgpd_ds_ids=[1, 2], cgpd_ds_ids=[1, 2], tmp_ds_ids=[1], dmp_ds_ids=[1],
        plot_best_elbo=True, settings_filter=good_settings_1,
        title="'Walk' dataset", filename="walk")
    process_datasets(vcgpd_ds_ids=[3, 4], cgpd_ds_ids=[3, 4], tmp_ds_ids=[2], dmp_ds_ids=[2],
        plot_best_elbo=True, settings_filter=good_settings_2,
        title="'Walk+wave' dataset", filename="walk_wave")

    process_datasets(vcgpd_ds_ids=[1, 2], cgpd_ds_ids=[1, 2], tmp_ds_ids=[1], dmp_ds_ids=[1],
        plot_best_elbo=True, settings_filter=good_settings_1, plot_tmp_elbo=False,
        title="'Walk' dataset", filename="walk_no_tmp_elbo")
    process_datasets(vcgpd_ds_ids=[3, 4], cgpd_ds_ids=[3, 4], tmp_ds_ids=[2], dmp_ds_ids=[2],
        plot_best_elbo=True, settings_filter=good_settings_2, plot_tmp_elbo=False,
        title="'Walk+wave' dataset", filename="walk_wave_no_tmp_elbo")
    
    #process_datasets(vcgpd_ds_ids=[3, 4], cgpd_ds_ids=[], tmp_ds_ids=[], dmp_ds_ids=[],
    #    plot_best_elbo=True,
    #    title="'Walk+wave' dataset", filename="walk_wave.pdf")
