import os
import time
import json
import logging
from six.moves import cPickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlibex.mlplot as plx
import ml.gptheano.vecgpdm.enums as enm
from ml.gptheano.vecgpdm.equations import list_of_nones
from ml.gptheano.vecgpdm.equations import matrix_of_nones
import validation.common as vc



pl = logging.getLogger(__name__)

titlesize = 20



def insert_nans(x=None, y=None, indexes=None):
    if x is None:
        x = np.arange(y.shape[0])
    xx = np.hstack([np.hstack([x[zi], [x[zi][-1]]]) for zi in indexes])
    yy = np.vstack([np.vstack([y[zi], [np.nan * y[zi][-1]]]) for zi in indexes])
    return xx, yy




def plot_latent_space(model):
    pl.info("Plotting the latent space. Coupling matrix Alpha: ")
    pl.info(model.ns.get_value(model.alpha))
    W = model.param.nparts  # number of parts
    plt.figure()
    for i in range(W):
        for j in range(W):
            plt.subplot(W, W, 1+i+j*W)
            plt.title("Latent space " + str(i), fontsize=titlesize)
            x_means = model.ns.get_value(model.x_means[j])
            plt.plot(x_means[:, 0], x_means[:, 1], '-', alpha=0.2)
            dyn_aug_z = model.ns.get_value(model.dyn_aug_z[j][i])
            plt.plot(dyn_aug_z[:, 0], dyn_aug_z[:, 1], 'o', markersize=15, markeredgewidth=2, fillstyle="none")
            lvm_aug_z = model.ns.get_value(model.lvm_aug_z[j])
            plt.plot(lvm_aug_z[:, 0], lvm_aug_z[:, 1], '+', markersize=15, markeredgewidth=2, fillstyle="none")
    plt.show()




def save_plot_latent_space(model, directory=None, prefix=None,
        plot_inducing_outputs=True,
        plot_sampled_trajectory=True):

    if prefix is None:
        prefix = "model"
    
    pl.info("Saving the latent space plots.")

    stats = {"alpha": model.ns.get_value(model.alpha).tolist()}
    if model.param.estimation_mode == enm.EstimationMode.ELBO:
        stats.update({"ELBO": model.get_elbo_value().tolist(),
                "dyn_ELBO": model.ns.evaluate(model.dyn_elbo).tolist(),
                "lvm_ELBO": model.ns.evaluate(model.lvm_elbo).tolist(),
                })
    else:
        stats.update({"Loglikelihood": model.get_loglikelihood_value().tolist(),})

    if directory is None:
        pl.info(json.dumps(stats))
    else:
        with open("{}/{}-stats.txt".format(directory, prefix), "w") as outfile:
            json.dump(stats, outfile)

    W = model.param.nparts  # number of parts
    
    # Sample the vector field
    xins = list_of_nones(W)
    step = 5
    for j in range(W):
        dyn_xtminus_means = model.ns.evaluate(model.dyn_xtminus_means[j])
        xins[j] = dyn_xtminus_means[0:len(dyn_xtminus_means):step]
        xins[j] = np.reshape(xins[j], [xins[j].shape[0], 2, -1])
    xouts = model.run_generative_dynamics(nsteps=2+1, startpoint=xins)
    xinout = [xouts[j][:, :, 0:2] for j in range(W)]

    # Run latent dynamics
    if plot_sampled_trajectory:
        N = model.param.data.N
        x_path = model.run_generative_dynamics(N)
    
    if model.param.estimation_mode == enm.EstimationMode.ELBO:
        # Augmenting mapping
        dyn_auginout = matrix_of_nones(W, W)
        for i in range(W):
            for j in range(W):
                dyn_auginout[j][i] = np.hstack([
                    model.pp_dyn_aug_z[j][i][:, np.newaxis, 0:2], 
                    model.pp_dyn_aug_z[j][i][:, np.newaxis, model.param.parts[j].Q:model.param.parts[j].Q+2],
                    model.pp_dyn_aug_u_means[j][i][:, np.newaxis, 0:2]])
            
    for i in range(W):
        for j in range(W):
            fig = plt.figure(figsize=(5, 5))
            x_means = model.ns.get_value(model.x_means[j])
            xx, yy = insert_nans(y=x_means, indexes=model.param.data.sequences_indexes)

            # Latent points
            plt.plot(yy[:, 0], yy[:, 1], '--b', alpha=0.2)

            # Vector field
            plx.plot_2nd_order_mapping_2d(xinout[j], alpha=0.4)

            # Sampled tralectory
            if plot_sampled_trajectory and i == j:
                plt.plot(x_path[j][:, 0], x_path[j][:, 1], "-k", alpha=1.0, linewidth=0.7) 
            
            if model.param.estimation_mode == enm.EstimationMode.ELBO:
                # Dyn inducing points
                dyn_aug_z = model.ns.get_value(model.dyn_aug_z[j][i])
                plt.plot(dyn_aug_z[:, 0], dyn_aug_z[:, 1], 
                        'ob', markersize=10, markeredgewidth=2, fillstyle="none")
                plt.plot(dyn_aug_z[:, model.param.parts[j].Q], dyn_aug_z[:, model.param.parts[j].Q+1], 
                        'ob', markersize=10, markeredgewidth=2, fillstyle="none")

                if plot_inducing_outputs:
                    plx.plot_2nd_order_mapping_2d(dyn_auginout[j][i], alpha=1, width=0.01)
                else:
                    plx.plot_arrows_2d(dyn_auginout[j][i][:, 0, :], dyn_auginout[j][i][:, 1, :], alpha=1, width=0.01)

                # Lvm inducing points
                lvm_aug_z = model.ns.get_value(model.lvm_aug_z[j])
                plt.plot(lvm_aug_z[:, 0], lvm_aug_z[:, 1], '+g', markersize=10, markeredgewidth=2, fillstyle="none")

            plt.title("Latent space. Part {}".format(j+1), fontsize=titlesize)
            
            if directory is None:
                plt.show()
            else:
                plt.savefig("{}/{}_latent_space_{}_to_{}_ip_{}.pdf".format(
                        directory, prefix, j, i, plot_inducing_outputs))
                plt.close(fig)



def save_plot_latent_vs_generated(model, directory=None, prefix=None):
    if prefix is None:
        prefix = "model"

    pl.info("Saving the latent space plots, latent vs generated.")
    W = model.param.nparts  # number of parts
    N = model.param.data.N
    x_path = model.run_generative_dynamics(N)
    for i in range(W):
        y_i = model.ns.get_value(model.x_means[i])
        xx, yy = insert_nans(y=y_i, indexes=model.param.data.sequences_indexes)

        fig = plt.figure(figsize=(5, 5))
        plt.plot(xx, yy, '--', linewidth=0.7, alpha=0.6)
        plt.gca().set_prop_cycle(None)
        plt.plot(x_path[i])
        plt.title("Latent trajectories. Part {}".format(i+1), fontsize=titlesize)
        if directory is None:
            plt.show()
        else:
            plt.savefig("{}/{}_latent_vs_generated_part_{}.pdf".format(
                    directory, prefix, i))
            plt.close(fig)



def plot_latent_vs_generated(model):
    N = model.modelparams.data.N
    W = model.modelparams.nparts  # number of parts
    x_path = model.run_generative_dynamics(N)
    plt.figure()
    for i in range(W):
        plt.subplot(W, 1, 1+i)
        x_means_i = model.ns.get_value(model.x_means[i])
        plt.plot(x_means_i, '--', linewidth=0.7, alpha=0.6)
        plt.gca().set_prop_cycle(None)
        plt.plot(x_path[i], linewidth=0.7)
    plt.show()



def save_plot_training_vs_generated(model, directory=None, prefix=None):
    if prefix is None:
        prefix = "model"

    pl.info("Saving the observed space plots, training vs generated.")
    N = model.param.data.N
    W = model.param.nparts  # number of parts
    x_path = model.run_generative_dynamics(N)
    y_path = model.lvm_map_to_observed(x_path)
    for i in range(W):
        y_i = model.param.parts[i].data.Y_value
        xx, yy = insert_nans(y=y_i, indexes=model.param.data.sequences_indexes)
        
        fig = plt.figure(figsize=(5, 5))
        plt.plot(xx, yy, '--', linewidth=0.5, alpha=0.6)
        plt.gca().set_prop_cycle(None)
        plt.plot(y_path[i], linewidth=0.7)
        plt.title("Training vs. generated. Part {}".format(i+1), fontsize=titlesize)

        if directory is None:
            plt.show()
        else:
            plt.savefig("{}/{}_training_vs_generated_part_{}_full.pdf".format(
                    directory, prefix, i))
            plt.close(fig)

        fig = plt.figure(figsize=(5, 5))
        plt.plot(xx, yy[:, :3], '--', linewidth=0.7, alpha=0.6)
        plt.gca().set_prop_cycle(None)
        plt.plot(y_path[i][:, :3], linewidth=0.7)
        plt.title("Training vs. generated. Part {}".format(i+1), fontsize=titlesize)

        if directory is None:
            plt.show()
        else:
            plt.savefig("{}/{}_training_vs_generated_part_{}_selected.pdf".format(
                    directory, prefix, i))
            plt.close(fig)

        



def plot_training_vs_generated(model):
    N = model.param.data.N
    W = model.param.nparts  # number of parts
    x_path = model.run_generative_dynamics(N)
    y_path = model.lvm_map_to_observed(x_path)
    plt.figure()
    for i in range(W):
        plt.subplot(W, 1, 1+i)
        y_i = model.param.parts[i].data.Y_value
        plt.plot(y_i, '-', alpha=0.2)
        plt.plot(y_path[i])
    plt.show()



def save_all_plots(model, directory, prefix):
    if directory is None:
        directory = "."
    if prefix is None:
        prefix = "model"
    
    save_plot_latent_space(model, directory, prefix, plot_inducing_outputs=True)
    save_plot_latent_space(model, directory, prefix, plot_inducing_outputs=False)
    save_plot_latent_vs_generated(model, directory, prefix)
    save_plot_training_vs_generated(model, directory, prefix)



class ModelPlotter(object):

    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        self.counter = 0
        
        
    def __call__(self, model):
        model.precalc_posterior_predictive()
        self._on_call(model)
        self.counter += 1


    def _on_call(self, model):
        prefix = "iter({})".format(self.counter)
        save_plot_latent_space(model, self.save_dir, prefix, plot_inducing_outputs=True)
        save_plot_latent_space(model, self.save_dir, prefix, plot_inducing_outputs=False)
        save_plot_latent_vs_generated(model, self.save_dir, prefix)
        save_plot_training_vs_generated(model, self.save_dir, prefix)


class MSEWriter(ModelPlotter):

    def _on_call(self, model):
        super(MSEWriter, self)._on_call(model)

        validation = np.array(model.param.data.Y_sequences)[0]
        print("Iteration counter: {}".format(self.counter))
        T_validation = len(validation)
        x_generated = model.run_generative_dynamics(T_validation)
        y_generated = model.lvm_map_to_observed(x_generated)
        predicted = np.hstack(y_generated)
        errors = vc.compute_errors(observed=validation, predicted=predicted)

        prefix = "iter({})".format(self.counter)
        with open("{}/{}-errors.txt".format(self.save_dir, prefix), "w") as outfile:
            json.dump(errors, outfile)


default_model_plotter = ModelPlotter()

    
