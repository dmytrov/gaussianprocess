import os
import numpy as np
import matplotlibex as plx
import ml.gptheano.vecgpdm.model as mdl
import numerical.numpytheano.theanopool as tp
import numerical.numpytheano as nt
import matplotlibex.mlplot as plx
import bvhrwroutines.bvhrwroutines as br



if __name__ == "__main__":
    #bvhfilename = "./2016.02.26_bjoern/IA-001_skeleton.bvh"  # move box
    bvhfilename = "./2016.02.26_bjoern/IC-001_skeleton.bvh"  # walking around
    #bvhfilename = "./2016.03.14_jumpy/1C_skeleton.bvh"  # walking around

    motion = br.BVH_Bridge()
    motion.read_BVH(bvhfilename)

    motion.add_part("head", ["neck", 
                             "spine2",
                             ])
    motion.add_part("upper", ["pelvis_spine1",
                              "spine1",
                              #"spine2",
                              #"neck",
                              "right_manubrium",
                              "right_clavicle",
                              "right_humerus",
                              "right_radius",
                              "left_manubrium",
                              "left_clavicle",
                              "left_humerus",
                              "left_radius",
                              ])
    motion.add_part("lower", ["pelvis", 
                              "pelvis_right_femur",
                              "right_femur_tibia",
                              "right_tibia_foot",
                              "pelvis_left_femur",
                              "left_femur_tibia",
                              "left_tibia_foot",
                              ])
    y, parts_IDs = motion.get_all_parts_data_and_IDs()

    # Walking in move box, bjoern
    #ylist = [y[45:90, :],
    #         y[585:634, :],
    #         y[1132:1165, :],
    #         y[1666:1710, :],
    #         y[2197:2235, :],]

    # walking around, bjoern
    #ylist = [
    #         y[21:182, :],  # ccw
    #         y[231:395, :],  # cw
    #         y[453:627, :],  # ccw
    #         y[683:851, :],  # cw
    #         y[901:1034, :],  # ccw
    #         y[1120:1293, :],  # cw
    #         ]

    # walking straight, bjoern
    ylist = [
             y[22:55, :],  # ccw
             y[231:263, :],  # cw
             y[476:486, :],  # ccw
             y[694:731, :],  # cw
             y[911:944, :],  # ccw
             y[1130:1161, :],  # cw
             ]

    # walking around, jumpy
    #ylist = [
    #         #y[57:228, :],  # ccw
    #         y[273:435, :],  # cw
    #         #y[506:627, :],  # ccw
    #         #y[687:877, :],  # cw
    #         #y[930:1102, :],  # ccw
    #         #y[1156:1330, :],  # cw
    #         ]

    #plx.plot_sequences(y)
    #exit()

    nparts = np.max(parts_IDs) + 1
    dyn_Ms = [16] * nparts
    lvm_Ms = [16] * nparts
    Qs = [3] * nparts
    
    directory = "movebox"
    if not os.path.exists(directory):
        os.makedirs(directory)
    nframes = 1000
    
    print("|=========== NumPy ==========|")
    ns = tp.NumpyVarPool()
    data = mdl.ModelData(ylist, ns=ns)
    params = mdl.ModelParam(data, Qs=Qs, parts_IDs=parts_IDs, dyn_Ms=dyn_Ms, lvm_Ms=lvm_Ms, ns=ns)
    model = mdl.VECGPDM(params, ns=ns)
    model.precalc_posterior_predictive()
    
    #mdl.plot_latent_space(model)
    mdl.save_plot_latent_space(model, directory, prefix="initial")

    filename = directory + "/initial_generated.bvh"
    x_path = model.run_generative_dynamics(nframes)
    y_path = model.lvm_map_to_observed(x_path)
    motion.set_all_parts_data(np.hstack(y_path))
    motion.write_BVH(filename)
    
    print("|=========== Theano ==========|")
    ns = tp.TheanoVarPool()
    data = mdl.ModelData(ylist, ns=ns)
    params = mdl.ModelParam(data, Qs=Qs, parts_IDs=parts_IDs, dyn_Ms=dyn_Ms, lvm_Ms=lvm_Ms, ns=ns)
    model = mdl.VECGPDM(params, ns=ns)
    
    mdl.save_plot_latent_space(model, directory, prefix="initial")
    mdl.optimize_blocked(model, niterations=3, maxiter=300, print_vars=True, save_directory=directory)
    
    filename = directory + "/final_generated.bvh"
    x_path = model.run_generative_dynamics(nframes)
    y_path = model.lvm_map_to_observed(x_path)
    motion.set_all_parts_data(np.hstack(y_path))
    motion.write_BVH(filename)

