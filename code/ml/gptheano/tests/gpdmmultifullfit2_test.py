import numpy as np
import matplotlibex as plx
import ml.gptheano.kernels as krn
import ml.gptheano.gpdmmultifullfit2 as gpdmmulti2

if __name__ == "__main__":
    t = np.linspace(0.0, 4*2*np.pi, num=100)
    y = np.vstack((5.0*np.sin(1*t+0.0), 5.0*np.sin(2*t+1.5),
                   3.1*np.sin(1*t+0.4), 3.1*np.sin(2*t+1.8),
                   0.1*np.sin(1*t+0.8), 0.1*np.sin(4*t+2.0),
                   0.1*np.sin(1*t+1.0), 0.1*np.sin(5*t+2.2))).T
    y = [y + 0.1*np.reshape(np.random.normal(size=y.size), y.shape),
         y + 0.1*np.reshape(np.random.normal(size=y.size), y.shape)]

    XVar = krn.MatrixVariable("X", np.identity(3))
    kernelLVM = lambda X1, X2: krn.SumKern([krn.RBFKern(X1, X2), krn.NoiseKern(X1, X2)])
    kernelDM = lambda X1, X2: krn.SumKern([krn.RBFKern(X1, X2), krn.LinearKern(X1, X2), krn.NoiseKern(X1, X2)])
    gp = gpdmmulti2.GPDMMulti2(kernelLVM=kernelLVM, kernelDM=kernelDM, Ysequences=y, Q=2, dynamicsOrder=2)

    plx.plot_sequence_variance_2d(gp.XVar.val, gp.predict)

    xgen, ygen = gp.generate_dynamics(500)
    plx.plot_sequence_variance_2d(xgen, gp.predict)



