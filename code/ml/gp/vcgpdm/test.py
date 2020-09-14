import numpy as np
import matplotlibex as plx
import ml.gp.vcgpdm.model as mdl

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

    np.random.seed(0)
    y = [y1 + 0.1*np.reshape(np.random.normal(size=y1.size), y1.shape),
         y2 + 0.1*np.reshape(np.random.normal(size=y2.size), y2.shape)]



    data = mdl.ModelData(y)
    params = mdl.ModelParams(data, Qs=[2, 2, 2], parts_IDs=[0]*8 + [1]*8 + [2]*8, M=10)
    model = mdl.VCGPDM(params)
    print(model.elbo())





