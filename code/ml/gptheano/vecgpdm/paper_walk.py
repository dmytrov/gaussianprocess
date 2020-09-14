import numpy as np
import matplotlibex as plx
import ml.gptheano.vecgpdm.model as mdl
import numerical.numpytheano.theanopool as tp
import numerical.numpytheano as nt
import matplotlibex.mlplot as plx
import bvhrwroutines.bvhrwroutines as br



if __name__ == "__main__":
    bvhfilename = "./walk/DanicaMapped_NeutralWalk02.bvh"

    motion = br.BVH_Bridge()
    motion.read_BVH(bvhfilename)

    motion.add_part("upper", ["Bip001_Spine1", 
                              "Bip001_Spine2",
                              "Bip001_Neck",
                              "Bip001_L_Clavicle",
                              "Bip001_L_UpperArm",
                              "Bip001_L_Forearm",
                              "Bip001_L_Hand",
                              "Bip001_L_Finger0",
                              "Bip001_L_Finger1",
                              "Bip001_R_Clavicle",
                              "Bip001_R_UpperArm",
                              "Bip001_R_Forearm",
                              "Bip001_R_Hand",
                              "Bip001_R_Finger0",
                              "Bip001_R_Finger1",
                              "Bip001_Head",
                              ])
    motion.add_part("lower", ["Bip001_Pelvis", 
                              "Bip001_Spine",
                              "Bip001_L_Thigh",
                              "Bip001_L_Calf", 
                              "Bip001_L_Foot", 
                              "Bip001_L_Toe0", 
                              "Bip001_R_Thigh", 
                              "Bip001_R_Calf", 
                              "Bip001_R_Foot", 
                              "Bip001_R_Toe0",
                              ])
    y, parts_IDs = motion.get_all_parts_data_and_IDs()
    print(y)
    print(parts_IDs)
    nparts = np.max(indexes)
    dyn_Ms = [8] * nparts
    lvm_Ms = [8] * nparts
    Qs = [3] * nparts
    
    directory = "walk"
    
    print("|=========== NumPy ==========|")
    ns = tp.NumpyVarPool()
    data = mdl.ModelData(y, ns=ns)
    params = mdl.ModelParam(data, Qs=Qs, parts_IDs=parts_IDs, dyn_Ms=dyn_Ms, lvm_M=lvm_Ms, ns=ns)
    model = mdl.VECGPDM(params, ns=ns)
    model.precalc_posterior_predictive()
    
    mdl.plot_latent_space(model)
    mdl.save_plot_latent_space(model, directory, prefix="initial")

    filename = directory + "/initial_generated.bvh"
    nframes = 200
    x_path = model.run_generative_dynamics(nframes)
    y_path = model.lvm_map_to_observed(x_path)
    motion.set_all_parts_data(np.hstack(y_path))
    motion.write_BVH(filename)
    
    print("|=========== Theano ==========|")
    ns = tp.TheanoVarPool()
    data = mdl.ModelData(y, ns=ns)
    params = mdl.ModelParam(data, Qs=Qs, parts_IDs=parts_IDs, dyn_Ms=dyn_Ms, lvm_M=lvm_Ms, ns=ns)
    model = mdl.VECGPDM(params, ns=ns)
    
    mdl.save_plot_latent_space(model, directory, prefix="initial")

    maxiter = 300
    print_vars = True
    model.optimize_by_tags(tags=set([mdl.VarTag.couplings]), maxiter=maxiter, print_vars=print_vars)
    model.optimize_by_tags(tags=set([mdl.VarTag.latent_x]), maxiter=maxiter, print_vars=print_vars)
    model.optimize_by_tags(tags=set([mdl.VarTag.kernel_params]), maxiter=maxiter, print_vars=print_vars)
    model.optimize_by_tags(tags=set([mdl.VarTag.couplings]), maxiter=maxiter, print_vars=print_vars)
    model.optimize_by_tags(tags=set([mdl.VarTag.augmenting_inputs]), maxiter=maxiter, print_vars=print_vars)
    model.optimize_by_tags(tags=set([mdl.VarTag.latent_x]), maxiter=maxiter, print_vars=print_vars)
    model.optimize_by_tags(tags=set([mdl.VarTag.latent_x, mdl.VarTag.augmenting_inputs]), maxiter=maxiter, print_vars=print_vars)
    model.optimize_by_tags(tags=set([mdl.VarTag.kernel_params]), maxiter=maxiter, print_vars=print_vars)
    model.optimize_by_tags(tags=set([mdl.VarTag.couplings]), maxiter=maxiter, print_vars=print_vars)

    model.precalc_posterior_predictive()

    mdl.save_plot_latent_space(model, directory, prefix="final")
    mdl.save_plot_latent_vs_generated(model, directory, prefix="final")
    mdl.save_plot_training_vs_generated(model, directory, prefix="final")


    filename = directory + "/final_generated.bvh"
    nframes = 200
    x_path = model.run_generative_dynamics(nframes)
    y_path = model.lvm_map_to_observed(x_path)
    motion.set_all_parts_data(np.hstack(y_path))
    motion.write_BVH(filename)

