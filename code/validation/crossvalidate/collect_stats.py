import os
import pickle
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import utils.parallel.runner as pr
import validation.crossvalidate.common as vcc
import validation.common as vc



class StatsReader(pr.JobRunner):
    
    def __init__(self):
        super(StatsReader, self).__init__(
                realpath=os.path.realpath(__file__),
                infostring="Collects statistics",
                createpathes=False)
        

    def create_params(self):
        return vcc.create_params(full=False)


    def is_must_run(self, taskparam):
        return True
 

    def on_run_task_complete(self, i):
        pass

    
    def pre_run(self):
        self.nparts = 3
        self.mse = [np.nan * np.zeros([40, 40, 5]) for i in range(self.nparts)]
        self.stats = vc.ErrorStatsReader()
        

    def run_task(self, param):
        dirname = os.path.join(self.args.dir, 
                self.taskparams.to_dir_name(param))
        try:
            self.stats.read_learned_errors_from_file(
                    os.path.join(dirname, "errors.pkl"), param=param)
        except:
            pass
        try:
            with open(os.path.join(dirname, "iter(3)-stats.txt"), "r") as outfile:
                additional = json.load(outfile)
                self.stats.errs[-1].update(additional)
        except:
            pass
            

    def post_run(self):
        print(self.args.dir)
        with open(os.path.join(self.args.dir, "combined_errors.pkl"), "wb") as filehandle:  
            pickle.dump(self.stats.errs, filehandle)

        #with open(os.path.join(self.args.dir, "combined_elbo_stats.pkl"), "wb") as filehandle:  
        #    pickle.dump(self.elbos, filehandle)
            
        exit()

        model_dataset_names = []
        best_elbo_infos = []
        best_elbo_means = []
        best_elbo_stds = []
        best_elbo_mse_means = []
        best_elbo_mse_stds = []
        best_mse_elbo_means = []
        best_mse_elbo_stds = []
        best_mse_infos = []
        best_mse_means = []
        best_mse_stds = []
        training_time_mean = []
        training_time_std = []

        datasetname = self.taskparams.param_range["dataset"][0]
        # All possible parameters range
        self.stats.params_range = [ 
                ("model", ["vcgpdm"]), 
                ("dataset", [datasetname]),
                ("mode", ["ELBO", "MAP"]),
                ("parts", [1, 3]),
                ("dyn", range(4, 11, 1)),
                ("lvm", range(4, 11, 1)),
                ("hold", range(0, 5)),
        ]

        # Select only vCGPDM, with full IPs range
        errs_axes = [
                ("model", ["vcgpdm"]), 
                ("dataset", [datasetname]),
                ("mode", ["ELBO", "MAP"]),
                ("parts", [1, 3]),
                ("dyn", range(4, 11, 1)),
                ("lvm", range(4, 11, 1)),
                ("hold", range(0, 5))]
        
        # vCGPDM data params sets
        params_set = [[
                    ("model", "vcgpdm"),
                    ("dataset", datasetname),
                    ("mode", "ELBO"),
                    ("parts", 1)
                ],[
                    ("model", "vcgpdm"),
                    ("dataset", datasetname),
                    ("mode", "ELBO"),
                    ("parts", 3)
                ],]

        for params in params_set:
            # Find a combination of IPs with best MSE 
            errs_value = self.stats.to_tensor(key="MSE", params_range=errs_axes)
            axes, data = vc.select_by(errs_axes, errs_value, params=params)
            MSE_means, MSE_stds = vc.mean_std(axes, data, alongs=["hold"])
            #print(MSE_means)
            i_best_MSE = np.nanargmin(MSE_means)  # best MSE index
            i, j = (i_best_MSE/len(axes[0][1]), i_best_MSE - len(axes[0][1]) * (i_best_MSE/len(axes[0][1])))
            print(params)
            print("Best parameters set: {} {}, {} {}".format(axes[0][0], axes[0][1][i], axes[1][0], axes[1][1][j]))
            
            errs_value = self.stats.to_tensor(key="ELBO", params_range=errs_axes)
            axes, data = vc.select_by(errs_axes, errs_value, params=params)
            ELBO_means, ELBO_std = vc.mean_std(axes, data, alongs=["hold"])
            i_best_ELBO = np.nanargmax(ELBO_means)  # best ELBO index
            
            #print(params)
            #print("MSE_means", MSE_means, MSE_stds)
            #print("ELBO_means", ELBO_means, ELBO_std)
            best_mse_mean = MSE_means.flatten()[i_best_MSE]
            best_mse_std = MSE_stds.flatten()[i_best_MSE]
            print("Best MSE: {} (+-{})".format(
                    best_mse_mean,
                    best_mse_std))  # best MSE and STD
            
            best_mse_elbo_mean = ELBO_means.flatten()[i_best_MSE]
            best_mse_elbo_std = ELBO_std.flatten()[i_best_MSE]
            print("Best MSE's ELBO: {} (+-{})".format(
                    best_mse_elbo_mean,
                    best_mse_elbo_std
                    ))  # best MSE and STD

            best_elbo_mean = ELBO_means.flatten()[i_best_ELBO]
            best_elbo_std = ELBO_std.flatten()[i_best_ELBO]
            print("Best ELBO: {} (+-{})".format(
                    best_elbo_mean,
                    best_elbo_std
                    ))  # best MSE and STD

            best_elbo_mse_mean = MSE_means.flatten()[i_best_ELBO]
            best_elbo_mse_std = MSE_stds.flatten()[i_best_ELBO]
            print("Best ELBO's MSE: {} (+-{})".format(
                    best_elbo_mse_mean,
                    best_elbo_mse_std
                    ))  # best MSE and STD
            
            model_dataset_names.append("N")
            best_elbo_infos.append("A")
            best_elbo_means.append(best_elbo_mean)
            best_elbo_stds.append(best_elbo_std)
            best_elbo_mse_means.append(best_elbo_mse_mean)
            best_elbo_mse_stds.append(best_elbo_mse_std)
            best_mse_elbo_means.append(best_mse_elbo_mean)
            best_mse_elbo_stds.append(best_mse_elbo_std)
            best_mse_infos.append("B")
            best_mse_means.append(best_mse_mean)
            best_mse_stds.append(best_mse_std)

            # Training time
            errs_value = self.stats.to_tensor(key="timing", params_range=errs_axes)
            axes, data = vc.select_by(errs_axes, errs_value, params=params)
            MSE_means, MSE_stds = vc.mean_std(axes, data, alongs=["hold", "dyn", "lvm"])
            print("Timing: {} (+-{})".format(MSE_means, MSE_stds))
            training_time_mean.append(MSE_means)
            training_time_std.append(MSE_stds)
            

        # Select only MAP CGPDM
        errs_axes = [
                ("model", ["vcgpdm"]), 
                ("dataset", [datasetname]),
                ("mode", ["MAP"]),
                ("parts", [1, 3]),
                ("hold", range(0, 5))]

        # MAP CGPDM data params sets
        params_set = [[
                    ("model", "vcgpdm"),
                    ("dataset", datasetname),
                    ("mode", "MAP"),
                    ("parts", 1)
                ],[
                    ("model", "vcgpdm"),
                    ("dataset", datasetname),
                    ("mode", "MAP"),
                    ("parts", 3)
                ],]

        for params in params_set:
            # Find a combination of IPs with best MSE 
            errs_value = self.stats.to_tensor(key="MSE", params_range=errs_axes)
            axes, data = vc.select_by(errs_axes, errs_value, params=params)
            #print(axes, data)
            MSE_mean, MSE_std = vc.mean_std(axes, data, alongs=["hold"])
            
            model_dataset_names.append("N")
            best_elbo_infos.append("A")
            best_elbo_means.append(0)
            best_elbo_stds.append(0)
            best_elbo_mse_means.append(0)
            best_elbo_mse_stds.append(0)
            best_mse_elbo_means.append(0)
            best_mse_elbo_stds.append(0)
            best_mse_infos.append("B")
            best_mse_means.append(MSE_mean)
            best_mse_stds.append(MSE_std)

            # Training time
            errs_value = self.stats.to_tensor(key="timing", params_range=errs_axes)
            axes, data = vc.select_by(errs_axes, errs_value, params=params)
            MSE_means, MSE_stds = vc.mean_std(axes, data, alongs=["hold"])
            print("Timing: {} (+-{})".format(MSE_means, MSE_stds))
            training_time_mean.append(MSE_means)
            training_time_std.append(MSE_stds)
        

        # TMP model
        errs_axes = [
                ("model", ["tmp"]), 
                ("dataset", [datasetname]),
                ("numprim", range(2, 11)),
                ("hold", range(0, 5))]
        
        params = [
                ("model", "tmp"),
                ("dataset", datasetname),
                ]
        
        errs_value = self.stats.to_tensor(key="MSE", params_range=errs_axes)
        axes, data = vc.select_by(errs_axes, errs_value, params=params)
        #print(axes, data)
        MSE_means, MSE_stds = vc.mean_std(axes, data, alongs=["hold"])
        #print(MSE_mean, MSE_std)
        
        i_best_MSE = np.nanargmin(MSE_means)
        best_mse_mean = MSE_means.flatten()[i_best_MSE]
        best_mse_std = MSE_stds.flatten()[i_best_MSE]
        print("Best MSE: {} (+-{})".format(
                best_mse_mean,
                best_mse_std))  # best MSE and STD
        
        model_dataset_names.append("N")
        best_elbo_infos.append("A")
        best_elbo_means.append(0)
        best_elbo_stds.append(0)
        best_elbo_mse_means.append(0)
        best_elbo_mse_stds.append(0)
        best_mse_elbo_means.append(0)
        best_mse_elbo_stds.append(0)
        best_mse_infos.append("B")
        best_mse_means.append(best_mse_mean)
        best_mse_stds.append(best_mse_std)

        # Training time
        errs_value = self.stats.to_tensor(key="timing", params_range=errs_axes)
        axes, data = vc.select_by(errs_axes, errs_value, params=params)
        MSE_means, MSE_stds = vc.mean_std(axes, data, alongs=["hold", "numprim"])
        print("Timing: {} (+-{})".format(MSE_means, MSE_stds))
        training_time_mean.append(MSE_means)
        training_time_std.append(MSE_stds)


        # DMP model
        errs_axes = [
                ("model", ["dmp"]), 
                ("dataset", [datasetname]),
                ("npsi", range(2, 51)),
                ("hold", range(0, 5))]
        
        params = [
                ("model", "dmp"),
                ("dataset", datasetname),
                ]
        
        errs_value = self.stats.to_tensor(key="MSE", params_range=errs_axes)
        axes, data = vc.select_by(errs_axes, errs_value, params=params)
        #print(axes, data)
        MSE_means, MSE_stds = vc.mean_std(axes, data, alongs=["hold"])
        i_best_MSE = np.nanargmin(MSE_means)
        best_mse_mean = MSE_means.flatten()[i_best_MSE]
        best_mse_std = MSE_stds.flatten()[i_best_MSE]
        print("Best MSE: {} (+-{})".format(
                best_mse_mean,
                best_mse_std))  # best MSE and STD

        # Training time
        errs_value = self.stats.to_tensor(key="timing", params_range=errs_axes)
        axes, data = vc.select_by(errs_axes, errs_value, params=params)
        MSE_means, MSE_stds = vc.mean_std(axes, data, alongs=["hold", "npsi"])
        print("Timing: {} (+-{})".format(MSE_means, MSE_stds))
        training_time_mean.append(MSE_means)
        training_time_std.append(MSE_stds)
        
        model_dataset_names.append("N")
        best_elbo_infos.append("A")
        best_elbo_means.append(0)
        best_elbo_stds.append(0)
        best_elbo_mse_means.append(0)
        best_elbo_mse_stds.append(0)
        best_mse_elbo_means.append(0)
        best_mse_elbo_stds.append(0)
        best_mse_infos.append("B")
        best_mse_means.append(best_mse_mean)
        best_mse_stds.append(best_mse_std)




        model_dataset_names = (
                "vGPDM", 
                "vCGPDM,\n3 parts", 
                "MAP GPDM", 
                "MAP CGPDM,\n3 parts", 
                "TMP", 
                "DMP")
        
        title = "'Passing an object' dataset"
        plot_best_elbo = True
        filename = datasetname + ".pdf"
        fig, ax1 = plt.subplots(figsize=(5, 5))
        #print(model_dataset_names)
        #print(best_elbo_infos)
        #print(best_elbo_means)
        #print(best_elbo_stds)
        #print(best_mse_infos)
        #print(best_mse_means)
        #print(best_mse_stds)
        #print(training_time_mean)
        #print(training_time_std)

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
            plt.savefig(filename)
            plt.close(fig)

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
        plt.savefig(datasetname + "-timing.pdf")
        plt.close(fig)

        
        # Dump stats to a file as a dict of dicts:
        # stats[model][param]
        stats = {ds_name:{} for ds_name in model_dataset_names}
        for i, dsname in enumerate(model_dataset_names):
            stats[dsname]["best_elbo_mean"] = best_elbo_means[i]
            stats[dsname]["best_elbo_std"] = best_elbo_stds[i]
            stats[dsname]["best_elbo_mse_mean"] = best_elbo_mse_means[i]
            stats[dsname]["best_elbo_mse_std"] = best_elbo_mse_stds[i]
            stats[dsname]["best_mse_elbo_mean"] = best_mse_elbo_means[i]
            stats[dsname]["best_mse_elbo_std"] = best_mse_elbo_stds[i]
            stats[dsname]["best_mse_mean"] = best_mse_means[i]
            stats[dsname]["best_mse_std"] = best_mse_stds[i]
            stats[dsname]["training_time_mean"] = training_time_mean[i]
            stats[dsname]["training_time_std"] = training_time_std[i]
        print(stats)
        with open(datasetname + "_stats.pkl", "wb") as filehandle:
            pickle.dump(stats, filehandle)
        

if __name__ == "__main__":
    StatsReader()
