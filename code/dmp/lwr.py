import matplotlib.pyplot as plt
import numpy as np
import matplotlibex as plx
import matplotlibex.mlplot as plx
import dmp.equations as deq


class LocalFunc(object):
    def __init__(self):
        pass

    def __call__(self, x):
        raise NotImplementedError()

    def plot(self, t, x):
        plt.plot(range(len(t)), self(x))


class RBFLocalFunc(LocalFunc):
    def __init__(self, center, sigmasqr):
        super(RBFLocalFunc, self).__init__()
        self.center = center
        self.sigmasqr = sigmasqr

    def __call__(self, x):
        return deq.RBF(self.center, self.sigmasqr, x)


class MisesBFLocalFunc(LocalFunc):
    def __init__(self, center, sigmasqr):
        super(MisesBFLocalFunc, self).__init__()
        self.center = center
        self.sigmasqr = sigmasqr

    def __call__(self, x):
        return deq.MisesBF(self.center, self.sigmasqr, x)




class LWR_1D(object):
    def __init__(self, x, y, npsi=20, regressor_func=None, local_funcs=None):
        self.x = x.copy()
        self.y = y.copy()

        if regressor_func is None:
            regressor_func = lambda x: np.ones_like(x)
        self.regressor_func = regressor_func
        if local_funcs is None:
            xmin = np.min(self.x)
            xmax = np.max(self.x)
            psi_cs = np.linspace(xmin, xmax, npsi)  # basis functions centres 
            psi_ls = np.zeros([npsi]) + 0.5 * ((xmax - xmin) / npsi)**2
            local_funcs = [RBFLocalFunc(center=psi_cs[i], sigmasqr=psi_ls[i]) for i in range(npsi)]
        self.local_funcs = local_funcs
        self.npsi = len(self.local_funcs)
        ksi = self.regressor_func(x)
        self.ws = np.zeros([self.npsi])
        for i in range(self.npsi):
            psi_i = self.get_psi_value(i, x)
            self.ws[i] = np.sum(ksi * psi_i * y) / np.sum(ksi * psi_i * ksi)   

    def get_psi_value(self, i, x):
        return self.local_funcs[i](x)

    def predict(self, xstar):
        if not isinstance(xstar, np.ndarray):
            xstar = np.array([xstar])
        ksi = self.regressor_func(xstar)  #[N]
        psi = [self.get_psi_value(i, xstar) for i in range(self.npsi)]  # [npsi][N]
        psi = np.vstack(psi).T  # [N, npsi]
        res = np.sum(psi * ksi[:, np.newaxis] * self.ws[np.newaxis, :], axis=1) / np.sum(psi, axis=1) 
        return res

    def plot_BFs(self, xstar):
        for i in range(self.npsi):
            plt.plot(xstar, self.get_psi_value(i, xstar))

class LWR(object):
    def __init__(self, x, y, npsi=20, regressor_func=None, local_funcs=None):
        self.D = y.shape[1]
        self.lwrs = []
        for i in range(self.D):
            lwr = LWR_1D(x,
                y[:, i],
                npsi,
                None if regressor_func is None else regressor_func[i],
                local_funcs)
            self.lwrs.append(lwr)

    def predict(self, xstar):
        return np.array([lwr.predict(xstar) for lwr in self.lwrs]).T

    def plot_BFs(self, xstar):
        self.lwrs[0].plot_BFs(xstar)

def test_LWR_1D():
    T = np.linspace(0.0, 11.0, 100)
    Y = np.sin(T) + 0.1 * np.random.normal(size=T.size)
    T = deq.x_of_t(T, 0.1)

    lwr = LWR_1D(T, Y)
    Ystar = lwr.predict(T)
    lwr.plot_BFs(T)
    plt.plot(T, Y)
    plt.plot(T, Ystar)
    plt.show()


def test_LWR():
    T = np.linspace(0.0, 11.0, 100)
    Y = np.vstack([np.sin(T) + 0.1 * np.random.normal(size=T.size),
                   np.cos(T) + 0.1 * np.random.normal(size=T.size)]).T
    T = deq.x_of_t(T, 0.1)

    lwr = LWR(T, Y, npsi=100)
    Ystar = lwr.predict(T)
    lwr.plot_BFs(T)
    plt.plot(T, Y)
    plt.plot(T, Ystar)
    plt.show()


if __name__ == "__main__":
    test_LWR()