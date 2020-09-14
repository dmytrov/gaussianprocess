import numpy as np
import matplotlibex as plx
import ml.gptheano.kernels as krn
import ml.gptheano.cgpdmmultifullfit as cgpdmmulti

if __name__ == "__main__":
    t = np.linspace(0.0, 4*2*np.pi, num=100)
    y1 = np.vstack((5.0*np.sin(1.0*t+0.0), 5.0*np.sin(2.0*t+1.5),
                    3.1*np.sin(1.0*t+0.4), 3.1*np.sin(2.0*t+1.8),
                    0.1*np.sin(1.6*t+0.4), 0.1*np.sin(0.6*t+1.8),
                    0.1*np.sin(1.6*t+0.4), 0.1*np.sin(0.6*t+1.8),

                    5.0*np.sin(1.0*t+0.0), 5.0*np.sin(2.0*t+1.5),
                    3.1*np.sin(1.0*t+0.4), 3.1*np.sin(2.0*t+1.8),
                    0.1*np.sin(1.6*t+0.4), 0.1*np.sin(0.6*t+1.8),
                    0.1*np.sin(1.6*t+0.4), 0.1*np.sin(0.6*t+1.8),

                    5.0*np.sin(1.3*t+0.0), 5.0*np.sin(2.6*t+1.5),
                    3.1*np.sin(1.3*t+0.4), 3.1*np.sin(2.6*t+1.8),
                    0.1*np.sin(1.8*t+0.4), 0.1*np.sin(0.1*t+1.8),
                    0.1*np.sin(1.8*t+0.4), 0.1*np.sin(0.1*t+1.8))).T

    y2 = np.vstack((5.0*np.sin(1.0*t+0.0), 5.0*np.sin(2.0*t+1.5),
                    3.1*np.sin(1.0*t+0.4), 3.1*np.sin(2.0*t+1.8),
                    0.1*np.sin(1.6*t+0.4), 0.1*np.sin(0.6*t+1.8),
                    0.1*np.sin(1.6*t+0.4), 0.1*np.sin(0.6*t+1.8),

                    5.0*np.sin(1.0*t+0.0), 5.0*np.sin(2.0*t+1.5),
                    3.1*np.sin(1.0*t+0.4), 3.1*np.sin(2.0*t+1.8),
                    0.1*np.sin(1.6*t+0.4), 0.1*np.sin(0.6*t+1.8),
                    0.1*np.sin(1.6*t+0.4), 0.1*np.sin(0.6*t+1.8),

                    5.0*np.sin(1.3*t+0.0), 5.0*np.sin(2.6*t+1.5),
                    3.1*np.sin(1.3*t+0.4), 3.1*np.sin(2.6*t+1.8),
                    0.1*np.sin(1.8*t+0.4), 0.1*np.sin(0.1*t+1.8),
                    0.1*np.sin(1.8*t+0.4), 0.1*np.sin(0.1*t+1.8))).T

    y = [y1 + 0.1*np.reshape(np.random.normal(size=y1.size), y1.shape),
         y2 + 0.1*np.reshape(np.random.normal(size=y2.size), y2.shape)]

    kernelLVM = lambda X1, X2: krn.SumKern([krn.RBFKern(X1, X2),
                                            krn.NoiseKern(X1, X2)])
    kernelLDM = lambda X1, X2: krn.SumKern([krn.RBFKern(X1, X2),
                                            #krn.LinearKern(X1, X2),
                                            krn.NoiseKern(X1, X2)])

    data = cgpdmmulti.Data(y)
    model_params = cgpdmmulti.ModelParams(data=data, Qs=[2, 2, 2], dynamics_order=2, parts_IDs=[0]*8 + [1]*8 + [2]*8,
                                          LVM_kernel_factory=kernelLVM, DM_kernel_factory=kernelLDM)

    gp = cgpdmmulti.CGPDMMulti(data=data, model_params=model_params)

    plx.plot_sequence_variance_2d(gp.XVar.val, gp.predict)

    xgen, ygen = gp.generate_dynamics(500)
    plx.plot_sequence_variance_2d(xgen, gp.predict)




