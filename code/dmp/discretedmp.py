import matplotlib.pyplot as plt
import numpy as np
import matplotlibex as plx
import numerical.numpytheano as nt
import numerical.numpyext.linalg as ntla
import dmp.equations as deq
import dmp.lwr as lwr
import dmp.canonical as canonical


class DMP(object):
    def __init__(self):
        pass


class DiscreteLocalRegressor(object):
    def __init__(self, g, y0):
        self.g = g
        self.y0 = y0

    def __call__(self, x):
        return x*(self.g-self.y0)


class DiscreteDMP(DMP):
    def __init__(self, npsi=10):
        super(DiscreteDMP, self).__init__()
        self.npsi = npsi
        self.dt = None
        self.alpha_z = 25.0
        self.beta_z = self.alpha_z / 4.0
        self.alpha_x = self.alpha_z / 3.0
        self.alpha_g = self.alpha_z / 2.0
        self.tau = 2.0
        self.phase_dynamics = None
        self.y = None
        self.y = None
        self.y0 = None
        self.g = None
        self.dy = None
        self.ddy = None
        self.lwrs = None
        self.local_regressor_func = None
        self.f_path = None
        self.y_state = None
        self.dy_state = None
        self.ddy_state = None
        self.y_path = None
        self.dy_path = None
        self.ddy_path = None


    def make_psi(self):
        tcentres = np.linspace(0.0, 1.0, self.npsi)
        xcentres = self.phase_dynamics.get_x_of_t(tcentres)
        widths = 0.3 / self.npsi**2 * np.abs(self.phase_dynamics.get_dx_of_t(tcentres))**2
        local_funcs = [lwr.RBFLocalFunc(center=xcentres[i], sigmasqr=widths[i]) for i in range(self.npsi)]
        return local_funcs


    def learn(self, ys):
        """
        :param y:  N*[T, D]  list of N training trajectories, each of D dimensions
        """
        if not isinstance(ys, list):
            ys = [ys]
        self.ytemplate = ys[0].copy()
        self.ys = ys
        self.y0 = np.mean(np.vstack([y[0, :] for y in ys]), axis=0)
        self.g = np.mean(np.vstack([y[-1, :] for y in ys]), axis=0)
        self.dt = 1.0 / self.ytemplate.shape[0]
        self.phase_dynamics = canonical.DiscreteCanonicalDynamics(self.alpha_x, self.tau, self.dt)
        self.x = []
        self.f_target = []
        #ntrials = len(self.ys)
        for i in range(len(self.ys)):
            y = self.ys[i]
            nsteps, D = y.shape
            dt = 1.0 / nsteps
            dy = np.diff(y, axis=0) / dt
            dy = np.vstack([dy[0, :], dy])
            ddy = np.diff(dy, axis=0) / dt
            ddy = np.vstack([ddy[0, :], ddy])
            ts, xs = canonical.DiscreteCanonicalDynamics(self.alpha_x, self.tau, dt).run(nsteps)
            f_target = deq.get_f_target(self.tau, self.alpha_z, self.beta_z, self.g[np.newaxis, :], y, dy, ddy)
            self.x.append(xs)
            self.f_target.append(f_target)

        self.x = np.hstack(self.x)
        self.f_target = np.vstack(self.f_target)
        # Approximate f_target with LWR
        self.local_funcs = self.make_psi()
        self.local_regressor_func = [DiscreteLocalRegressor(self.g[d], self.y0[d]) for d in range(D)]
        self.lwrs = lwr.LWR(x=self.x, y=self.f_target,
            regressor_func=self.local_regressor_func,
            local_funcs=self.local_funcs)


    def plot(self):
        nsteps, D = self.ytemplate.shape
        self.phase_dynamics.reset()
        ts, xs = self.phase_dynamics.run(nsteps)
        plt.plot(xs)
        for lf in self.local_funcs:
            lf.plot(ts, xs)
        f = self.lwrs.predict(xs)
        z = 3.0 / np.max(self.f_target[:, 0])
        plt.plot(z * self.f_target[:, 0])
        plt.plot(z * f[:, 0])


    def reset(self):
        self.phase_dynamics.reset()
        self.y_state = self.y0.copy()
        self.dy_state = 0.0 * self.y0
        self.ddy_state = 0.0 * self.y0
        self.f_path = np.array([])
        self.y_path = np.array([])
        self.dy_path = np.array([])
        self.ddy_path = np.array([])


    def run(self, nsteps=None):
        if nsteps is None:
            nsteps = self.ytemplate.shape[0]
        nDMPs = self.ytemplate.shape[1]
        self.f_path = np.zeros([nsteps, nDMPs])
        self.y_path = np.zeros([nsteps, nDMPs])
        self.dy_path = np.zeros([nsteps, nDMPs])
        self.ddy_path = np.zeros([nsteps, nDMPs])
        ts, xs = self.phase_dynamics.run(nsteps)
        for i in range(nsteps):
            f_x = self.lwrs.predict(xs[i])
            self.ddy_state = deq.get_ddy(self.tau, self.alpha_z, self.beta_z, self.g, self.y_state, self.dy_state, f_x)[0]
            self.dy_state += self.ddy_state * self.dt
            self.y_state += self.dy_state * self.dt
            self.f_path[i] = f_x
            self.y_path[i] = self.y_state
            self.dy_path[i] = self.dy_state
            self.ddy_path[i] = self.ddy_state




def test_DiscreteDMP():
    N1 = 200
    T = np.linspace(0.0, 20.0, N1)
    Y1 = np.vstack([5*np.cos(T) * T * (T[-1]-T) / (T[-1]/2)**2,
                    3*np.cos(T)]).T
    Y1 += np.reshape(0.001 * np.random.normal(size=Y1.size), Y1.shape)
    N2 = 300
    T = np.linspace(0.0, 20.0, N2)
    Y2 = np.vstack([5*np.cos(T) * T * (T[-1]-T) / (T[-1]/2)**2,
                    3*np.cos(T)]).T
    Y2 += np.reshape(0.001 * np.random.normal(size=Y2.size), Y2.shape)
    Y = [Y1, Y2]
    dmp = DiscreteDMP(npsi=50)
    dmp.learn(Y)
    dmp.reset()
    dmp.run(2 * N1)
    plt.subplot(2, 1, 1)
    dmp.plot()
    plt.subplot(2, 1, 2)
    plt.plot(Y1[:, 0], label="Y")
    plt.plot(dmp.y_path[:, 0], label="y_path")
    plt.plot(0.001*dmp.f_path[:, 0], label="f_path")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    test_DiscreteDMP()
