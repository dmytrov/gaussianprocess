import numpy as np
import scipy as sp
import numerical.numpytheano as nt
import numpytheano.varpool as vp
import matplotlib.pyplot as plt


def fit_main_frequency(x, w, a, p):
    """
    Fits signal x with a sine with initial frequency w, ampliture a, phase p
    :param x: [N, D] D channels of the signal
    :param w: (scalar) initial frequency
    :param a : [D] initial amplitude
    :param p: [D] initial phase of the signal
    :return: frequency (scalar), amplitude [D], phase [D] of the signal
    """
    N, D = x.shape
    ns = vp.TheanoVarPool()
    wvar = ns.scalar("frequency", value=w)
    avar = ns.vector("amplitudes", value=a)
    pvar = ns.vector("phases", value=p)
    xfit = avar * ns.cos(wvar * 2.0 * np.pi * np.linspace(0.0, 1.0, N)[:, np.newaxis] + pvar)
    err = ns.sum((x - xfit)**2)
    f_df = ns.make_function_and_gradient(err, args=all)
    vp.optimize_bfgs_l(f_df)
    wopt = ns.get_value(wvar)
    aopt = ns.get_value(avar)
    popt = ns.get_value(pvar)
    return wopt, aopt, popt

def estimate_main_frequency(x):
    """
    Estimates amplitude, phase and main frequency of the multichannel signal x
    :param x: [N, D] D channels of the signal
    :return: frequency (scalar), amplitude [D], phase [D] of the signal
    """
    if x.ndim < 2:
        x = x[:, np.newaxis]
    assert x.ndim == 2
    N, D = x.shape

    fx = np.vstack([sp.fft(xi)[:int(N/2)] for xi in x.T]).T
    ax = 2.0 /N * np.vstack([np.abs(fxi) for fxi in fx.T]).T
    axsum = np.sum(ax, axis=1)
    wmax = np.argmax(axsum)  # T*Hz, base frequency
    amax = ax[wmax]  # amplitude of base frequency
    phasex = np.angle(fx)
    phasemax = phasex[wmax]  # phase of base frequency
    wmax = float(wmax)
    wopt, aopt, popt = fit_main_frequency(x, wmax, amax, phasemax)
    return wopt, aopt, popt


def test_estimate_main_frequency():
    dt = 1.0 / 1000
    T = 2.0  # sec
    t = dt * np.arange(int(T/dt))
    f1 = 10.5  # Hz
    x1 = 3.0*np.cos(f1 * 2.0 * np.pi * t + 0.1*3.14)
    f2 = f1  # Hz
    x2 = 2.5 * np.cos(f2 * 2.0 * np.pi * t + 0.2*3.14)
    f3 = f1  # Hz
    x3 = 1.5 * np.sin(f3 * 2.0 * np.pi * t + 0.3*3.14)
    x = np.vstack([x1, x2, x3]).T

    wopt, aopt, popt = estimate_main_frequency(x)
    print("Frequency: {}\nAmplitude: {}\nPhase: {}".format(wopt / T, aopt, popt))
    xfit = aopt * np.cos(wopt * 2.0 * np.pi * np.linspace(0.0, 1.0, len(t))[:, np.newaxis] + popt)
    plt.plot(t, x[:, 0])
    plt.plot(t, xfit[:, 0])
    plt.show()



if __name__ == "__main__":
    test_estimate_main_frequency()
