import matplotlib.pyplot as plt
import numpy as np
import matplotlibex as plx
import numerical.numpytheano as nt
import numerical.numpyext.linalg as ntla


"""
Dynamic Motor Primitives by Schaal
See:
[1] Schaal (2013) - From dynamic movement primitives to associative skill memories
[2] Schaal (2013) - Dynamical Movement Primitives Learning Attractor Models for Motor Behaviors
"""


def get_ddy(tau, alpha_z, beta_z, g, y, dy, f):
    """
    Equation [1]
    :param tau:  time constant
    """
    return (1.0 / tau**2) * (alpha_z * (beta_z * (g - y) - tau*dy) + f)


def get_f_target(tau, alpha_z, beta_z, g, y, dy, ddy):
    return tau**2 * ddy - alpha_z * (beta_z *(g - y) - tau * dy)


def get_z_prime(tau, alpha_z, beta_z, g, y, z, f):
    """
    Equation [2](2.1.1)
    :param tau:  time constant
    """
    return (1.0 / tau) * (alpha_z * (beta_z * (g - y) - z) + f)


def get_y_prime(tau, z):
    """
    Equation [2](2.1.2)
    :param tau:  time constant
    """
    return (1.0 / tau) * z


def get_x_prime(tau, z):
    """
    Equation [2](2.2)
    :param tau:  time constant
    """
    return (1.0 / tau) * z


def get_f_of_x(x, ws, psis, g, y_0, ns=nt.NumpyLinalg):
    """
    Equation [2](2.3)
    :param x: internal time
    :param ws: weights
    :param psis: basis functions
    :param g: goal
    :param y_0: initial state y_0 = y(t=0)
    """
    N = len(psis)
    psivals = ns.concatenate([psi_i(x, ns=ns) for psi_i in psis])
    return x * (g-y_0) * np.sum(psivals * ws) / ns.sum(psivals)


def get_psi_i(x, sigmasqr, c, ns=nt.NumpyLinalg):
    return  ns.exp(-0.5 * (x-c)**2 / sigmasqr)


def RBF(center, sigmasqr, X):
    """
    Radial Basis Function
    :param center:
    :param sigmasqr: 
    :param X:  [N, D] - N points of D dimensionality
    """
    sigmasqrinv = 1.0 / sigmasqr
    return np.exp(-0.5 * (X-center)**2 * sigmasqrinv)


def x_of_t(T, tau):
    return np.exp(-tau * T)


def ksi_of_t_discrete(x_func, T, y0, g):
    """
    Local regressor.
    Discrete system
    """
    return x_func(T) * (g - y0)


def ksi_of_t_rhythmic(T, r):
    """
    Local regressor.
    Rhythmic system
    """
    return 0 * T + r


def plot_psi(T, psi_cs, psi_ls):
    npsi = len(psi_cs)
    psi = [RBF(psi_cs[i], psi_ls[i], T) for i in range(npsi)]  # [npsi][N]
    for i in range(npsi):
        plt.plot(psi[i])


def MisesBF(center, sigmasqr, X):
    """
    Von Mises Basis Function
    :param center:
    :param sigmasqr: 
    :param X:  [N, D] - N points of D dimensionality
    """
    sigmasqrinv = 1.0 / sigmasqr
    return np.exp((np.cos(X-center)-1) * sigmasqrinv)



if __name__ == "__main__":
    pass
