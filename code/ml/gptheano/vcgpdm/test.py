import numpy as np
import matplotlibex as plx
import ml.gptheano.vcgpdm.model as mdl
import numerical.numpytheano as nt
import matplotlibex.mlplot as plx

if __name__ == "__main__":
    t = np.linspace(0.0, 10.1*2*np.pi, num=1000)
    y1 = np.vstack((5.0*np.sin(1.0*t+0.0), 5.0*np.sin(1.0*t+1.5),
                    4.1*np.sin(1.0*t+0.4), 4.1*np.sin(1.0*t+1.8),
                    3.1*np.sin(1.0*t+0.4), 3.1*np.sin(1.0*t+1.8),
                    2.1*np.sin(1.0*t+0.4), 2.1*np.sin(1.0*t+1.8),

                    #5.0*np.sin(1.0*t+0.0), 5.0*np.sin(2.0*t+1.5),
                    #3.1*np.sin(1.0*t+0.4), 3.1*np.sin(2.0*t+1.8),
                    #0.1*np.sin(1.6*t+0.4), 0.1*np.sin(0.6*t+1.8),
                    #0.1*np.sin(1.6*t+0.4), 0.1*np.sin(0.6*t+1.8),

                    5.0*np.sin(1.3*t+0.0), 5.0*np.sin(2.6*t+3.5),
                    4.1*np.sin(1.3*t+0.4), 4.1*np.sin(2.6*t+2.8),
                    3.1*np.sin(1.3*t+0.4), 3.1*np.sin(2.6*t+3.8),
                    2.1*np.sin(1.3*t+0.4), 2.1*np.sin(2.6*t+4.8)
                    )).T

    y2 = np.vstack((5.0*np.sin(1.0*t+0.0), 5.0*np.sin(2.0*t+1.5),
                    4.1*np.sin(1.0*t+0.4), 4.1*np.sin(2.0*t+1.8),
                    3.1*np.sin(1.0*t+0.4), 3.1*np.sin(2.0*t+1.8),
                    2.1*np.sin(1.0*t+0.4), 2.1*np.sin(2.0*t+1.8),

                    #5.0*np.sin(1.0*t+0.0), 5.0*np.sin(2.0*t+1.5),
                    #3.1*np.sin(1.0*t+0.4), 3.1*np.sin(2.0*t+1.8),
                    #0.1*np.sin(1.6*t+0.4), 0.1*np.sin(0.6*t+1.8),
                    #0.1*np.sin(1.6*t+0.4), 0.1*np.sin(0.6*t+1.8),

                    #5.0*np.sin(1.3*t+0.0), 5.0*np.sin(2.6*t+1.5),
                    #3.1*np.sin(1.3*t+0.4), 3.1*np.sin(2.6*t+1.8),
                    #0.1*np.sin(1.8*t+0.4), 0.1*np.sin(0.1*t+1.8),
                    #0.1*np.sin(1.8*t+0.4), 0.1*np.sin(0.1*t+1.8)
                    )).T

    np.random.seed(0)
    y = [y1 + 0.01*np.reshape(np.random.normal(size=y1.size), y1.shape),
         #y2 + 0.1*np.reshape(np.random.normal(size=y2.size), y2.shape)
         ]
    M = 8

    #plx.plot_sequences(y1)
    print("|=========== NumPy ==========|")
    ns = nt.NumpyLinalg
    data = mdl.ModelData(y, ns=ns)
    #params = mdl.ModelParams(data, Qs=[2], parts_IDs=[0]*8, M=20, ns=ns)
    params = mdl.ModelParams(data, Qs=[2, 2], parts_IDs=[0]*8 + [1]*8, M=M, ns=ns)
    model = mdl.VCGPDM(params, ns=ns)

    model.precalc_posterior_predictive()
    x_pathes, y_pathes = model.run_generative_mode()
    plx.plot_sequence2d(x_pathes[0], "X path")
    plx.plot_sequences(y_pathes[0], " Y path")
    

    #xminust_stars = model.give_first_xminust_stars()
    #for i in range(10):
    #    xtstars = model.dyn_posterior_predictive(xminust_stars)
    #    print(xtstars)
    #    ystars = model.lvm_posterior_predictive(xtstars)
    #    print(ystars)
    #    xminust_stars = model.update_xminust_stars(xminust_stars, xtstars)

    print("|=========== Theano ==========|")
    #ns = nt.NumpyLinalg
    ns = nt.TheanoLinalg
    data = mdl.ModelData(y, ns=ns)
    params = mdl.ModelParams(data, Qs=[2, 2], parts_IDs=[0]*8 + [1]*8, M=M, ns=ns)
    model = mdl.VCGPDM(params, ns=ns)

    model.precalc_posterior_predictive()
    plx.plot_sequence2d(model.parts[0].pp_dyn_aug_in, title="Dynamics augmenting inputs")
    x_pathes, y_pathes = model.run_generative_mode()
    plx.plot_sequence2d(x_pathes[0], "X path")
    plx.plot_sequences(y_pathes[0], " Y path")

    
    print("|================================|")
    print("Coupling matrix: ", model.get_coupling_matrix_vales())        
    model.optimize_kernel_params(maxiter=10)
    print("Coupling matrix: ", model.get_coupling_matrix_vales())        

    model.precalc_posterior_predictive()
    plx.plot_sequence2d(model.parts[0].pp_dyn_aug_in, title="Dynamics augmenting inputs")
    x_pathes, y_pathes = model.run_generative_mode()
    plx.plot_sequence2d(x_pathes[0], "X path")
    plx.plot_sequences(y_pathes[0], " Y path")
    

    print("|================================|")
    model.optimize_all(maxiter=100)
    print("Coupling matrix: ", model.get_coupling_matrix_vales())        
    
    model.precalc_posterior_predictive()
    plx.plot_sequence2d(model.parts[0].pp_dyn_aug_in, title="Dynamics augmenting inputs")
    x_pathes, y_pathes = model.run_generative_mode()
    plx.plot_sequence2d(x_pathes[0], "X path")
    plx.plot_sequences(y_pathes[0], " Y path")
    
    print("|================================|")
    print("Coupling matrix: ", model.get_coupling_matrix_vales())        
    model.optimize_kernel_params(maxiter=10)
    print("Coupling matrix: ", model.get_coupling_matrix_vales())        

    model.precalc_posterior_predictive()
    plx.plot_sequence2d(model.parts[0].pp_dyn_aug_in, title="Dynamics augmenting inputs")
    x_pathes, y_pathes = model.run_generative_mode()
    plx.plot_sequence2d(x_pathes[0], "X path")
    plx.plot_sequences(y_pathes[0], " Y path")






