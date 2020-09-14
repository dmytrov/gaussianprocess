import numpy as np
import matplotlibex as plx
import ml.gptheano.kernels as krn

import ml.gptheano.gplvmfullfit as gplvm

if __name__ == "__main__":
    t = np.linspace(0.0, 3*2*np.pi, num=300)
    y = np.vstack((3*np.sin(1*t+0.0), 3*np.sin(2*t+1.5),
                   1*np.sin(1*t+0.4), 1*np.sin(3*t+1.8),
                   1*np.sin(1*t+0.8), 1*np.sin(4*t+2.0),
                   1*np.sin(1*t+1.0), 1*np.sin(5*t+2.2))).T
    y = y + 0.1*np.reshape(np.random.normal(size=y.size), y.shape)

    XVar = krn.MatrixVariable("X", np.identity(3))
    krbfnoise = krn.SumKernel([krn.RBFKernel(XVar, XVar), krn.NoiseKernel(XVar, XVar)])
    gp = gplvm.GPLVM(y, 2, krbfnoise)
    print("##")
    plx.plot_sequence_variance_2d(gp.XVar.val, gp.predict)

