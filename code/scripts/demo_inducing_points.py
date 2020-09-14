import numpy as np
import matplotlib.pyplot as plt
import matplotlibex as plx


def kern_RBF(x1, x2):
    return np.exp(-20.5*np.sum((x1-x2)**2))

def gramm_matrix(f_kern, X, Xprime=None):
    if Xprime is None:
        Xprime = X
    N = X.shape[0]
    K = np.zeros([N, N])
    K = np.array([[f_kern(xi, xj) for xj in Xprime] for xi in X])
    return K

def gp_sample(mean, cov):
    return np.random.multivariate_normal(mean=mean, cov=cov)

def gp_conditional_pdf(f_kern, Xstar, X, Y, f_mean=None):
    if f_mean is None:
        f_mean = lambda x: 0
    Kxx = gramm_matrix(f_kern, X) + 1.0e-2 * np.diag(np.ones(X.shape[0]))
    Kxstarx = gramm_matrix(f_kern, Xstar, X)
    Ystar_mean = f_mean(Xstar) + Kxstarx.dot(np.linalg.inv(Kxx).dot(Y))
    Ystar_cov = gramm_matrix(f_kern, Xstar) - Kxstarx.dot(np.linalg.inv(Kxx).dot(Kxstarx.T))
    return Ystar_mean, Ystar_cov


if __name__ == "__main__":
    np.random.seed(5)
    N = 100
    D = 1
    X = np.reshape(np.random.uniform(size=N*D), [N, D])
    X = np.sort(X, axis=0)
    K = gramm_matrix(f_kern=kern_RBF, X=X)
    Y = gp_sample(mean=np.zeros(N), cov=K)
    np.random.seed(12)
    istart = [10, 41, 95, 70, 80]
    a = np.array(list(set(range(N)) - set(istart)))
    inds = np.concatenate([istart, np.random.choice(a=a, size=N-len(istart), replace=False)])
    Ninds = [0, 1, 2, 3, 4, 5, 10, N]
    for Nind in Ninds:
        ind = np.sort(inds[:Nind])
        Xind = X[ind]
        Yind = Y[ind]
        Xstar = np.linspace(0, 1.0, N)
        Ystar_mean, Ystar_cov = gp_conditional_pdf(f_kern=kern_RBF, Xstar=Xstar, X=Xind, Y=Yind)
        Ystar_err = np.diag(Ystar_cov) + 0.16
        fig = plt.figure(figsize=(5, 5))
        plt.title("#IP={}".format(Nind))
        plt.plot(X, Y + 0.1*np.random.normal(size=N), "+r")
        plt.plot(Xstar, Ystar_mean)
        plt.plot(Xind, Yind, "ok")
        plt.fill(np.concatenate([Xstar, Xstar[::-1]]), 
                np.concatenate([Ystar_mean - Ystar_err, (Ystar_mean + Ystar_err)[::-1]]), alpha=.2)
        plt.ylim([-3, 1.5])
        plt.savefig("./IP({}).pdf".format(Nind))
        plt.close(fig)
        