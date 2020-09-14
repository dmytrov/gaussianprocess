import matplotlib.pyplot as plt
import numpy as np
import matplotlibex as plx
import numerical.numpytheano as nt
import numerical.numpyext.linalg as ntla



class CanonicalDynamics(object):
    def __init__(self, tau, dt):
        self.tau = tau
        self.dt = dt
        self.t0 = 0.0  # initial state
        self.x0 = 1.0  # initial state
        self.t = 0.0  # current state
        self.x = 1.0  # current state
        self.reset()

    def reset(self):
        self.t = self.t0
        self.x = self.x0

    def get_x_of_t(self, t):
        raise NotImplementedError()

    def run(self, nsteps=24):
        ts = self.t + self.dt * np.arange(1, nsteps+1)
        xs = self.get_x_of_t(ts)
        self.t = ts[-1]
        self.x = xs[-1]
        return ts, xs


class DiscreteCanonicalDynamics(CanonicalDynamics):
    def __init__(self, alpha_x=25.0/3.0, tau=10.0, dt=1.0/24.0):
        super(DiscreteCanonicalDynamics, self).__init__(tau, dt)
        self.alpha_x = alpha_x

    def get_x_of_t(self, t):
        # Analytic solution
        x_path = self.x0 * np.exp(-self.alpha_x * t / self.tau)
        return x_path

    def get_dx_of_t(self, t):
        # Analytic solution
        dx = - self.x0 * self.alpha_x / self.tau * np.exp(-self.alpha_x * t / self.tau)
        return dx

    def get_t_of_x(self, x):
        # Inverse analytic solution
        return - self.tau * np.log(x / self.x0) / self.alpha_x




class RhythmicCanonicalDynamics(CanonicalDynamics):
    def __init__(self, tau=1.0, dt=1.0/24.0):
        super(RhythmicCanonicalDynamics, self).__init__(tau, dt)

    def _get_period(self):
        return 2.0 * np.pi / self.tau

    def get_x_of_t(self, t):
        # Phase at time t
        # x = [0..2*pi] for t = [0..1]
        # Analytic solution
        phase = t * self._get_period()
        return np.mod(phase, 2.0 * np.pi)

    def get_dx_of_t(self, t):
        # Analytic solution
        return self._get_period() + np.zeros_like(t)

    def get_t_of_x(self, x):
        # Inverse analytic solution
        return x / self._get_period()



def test_DiscreteCanonicalDynamics():
    dcd = DiscreteCanonicalDynamics()
    plt.plot(*dcd.run())
    plt.plot(*dcd.run())
    plt.plot(*dcd.run())
    plt.show()

def test_RhythmicCanonicalDynamics():
    dcd = RhythmicCanonicalDynamics()
    plt.plot(*dcd.run())
    plt.plot(*dcd.run())
    plt.plot(*dcd.run())
    plt.show()


if __name__ == "__main__":
    #test_DiscreteCanonicalDynamics()
    test_RhythmicCanonicalDynamics()
