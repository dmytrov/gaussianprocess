import matplotlib.pyplot as plt
import numpy as np
import matplotlibex as plx
import numerical.numpytheano as nt
import numerical.numpyext.linalg as ntla
import dmp.equations as deq
import dmp.lwr as lwr
import dmp.canonical as canonical


class DMP(object):
    def __init__(self, npsi=10):
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
        raise NotImplementedError()


    def learn(self, ys):
        """
        :param y:  [T, D]  training trajectory
        """
        raise NotImplementedError()


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


