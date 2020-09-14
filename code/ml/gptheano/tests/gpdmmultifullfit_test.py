import numpy as np
import matplotlibex as plx
import ml.gptheano.kernels as krn
import ml.gptheano.gpdmmultifullfit as gpdmmulti

if __name__ == "__main__":
    t = np.linspace(0.0, 10*2*np.pi, num=300)
    y = np.vstack((3*np.sin(1*t+0.0), 3*np.sin(2*t+1.5),
                   0.1*np.sin(1*t+0.4), 0.1*np.sin(3*t+1.8),
                   0.1*np.sin(1*t+0.8), 1.1*np.sin(4*t+2.0),
                   0.1*np.sin(1*t+1.0), 0.1*np.sin(5*t+2.2))).T
    y = y + 0.2*np.reshape(np.random.normal(size=y.size), y.shape)

    XVar = krn.MatrixVariable("X", np.identity(3))
    krbfnoise = krn.SumKernel([krn.RBFKernel(XVar, XVar), krn.NoiseKernel(XVar, XVar)])
    krbfnoise2order = krn.SumMultiSequenceKernel([krn.RBFSecondOrderMultiSequenceKernel(XVar, XVar),
                                                  krn.NoiseSecondOrderMultiSequenceKernel(XVar, XVar)])
    gp = gpdmmulti.GPDMMulti(dynamicsOrder=2, Ysequences=y, Q=2, kernelLVM=krbfnoise, kernelDM=krbfnoise2order)
    print("##")
    plx.plot_sequence_variance_2d(gp.XVar.val, gp.predict)

    xgen, ygen = gp.generate_dynamics(300)
    plx.plot_sequence_variance_2d(xgen, gp.predict)


