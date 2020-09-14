import numpy as np
import matplotlib.pyplot as plt
import matplotlibex as plx
import numerical.numpyext.linalg as ntlinalg


inv = ntlinalg.cholesky_inv_jitter

def kern_RBF(x1, x2):
    return np.exp(-40.5*np.sum((x1-x2)**2))

def gramm_matrix(f_kern, X, Xprime=None):
    if Xprime is None:
        Xprime = X
    N = X.shape[0]
    K = np.zeros([N, N])
    K = np.array([[f_kern(xi, xj) for xj in Xprime] for xi in X])
    return K

def gp_sample(mean, cov):
    return np.random.multivariate_normal(mean=mean, cov=cov)

def gp_conditional_pdf(f_kern, sigmasqr, Xstar, X, Y, f_mean=None):
    if f_mean is None:
        f_mean = lambda x: 0
    Kxx = gramm_matrix(f_kern, X) + sigmasqr * np.identity(X.shape[0])
    Kxstarx = gramm_matrix(f_kern, Xstar, X)
    Ystar_mean = f_mean(Xstar) + Kxstarx.dot(inv(Kxx).dot(Y))
    Ystar_cov = gramm_matrix(f_kern, Xstar) - Kxstarx.dot(inv(Kxx).dot(Kxstarx.T))
    return Ystar_mean, Ystar_cov


if __name__ == "__main__":
    np.random.seed(6)
    N = 100
    D = 1
    X = np.reshape(np.random.uniform(size=N*D), [N, D])
    X = np.sort(X, axis=0)
    K = gramm_matrix(f_kern=kern_RBF, X=X)
    F = gp_sample(mean=np.zeros(N), cov=K)
    sigmasqr = 0.01
    Y = F + np.sqrt(sigmasqr) * np.random.normal(size=N)
    Ffiltered = (inv(inv(K) + np.identity(N) * (1.0/sigmasqr))).dot(Y) / sigmasqr
    Ystar_mean, Ystar_cov = gp_conditional_pdf(f_kern=kern_RBF, sigmasqr=sigmasqr, Xstar=X, X=X, Y=Y)
    
    fig = plt.figure(figsize=(5, 5))
    plt.plot(X, Y, "xk")
    plt.plot(X, F, "--k")
    plt.plot(X, Ffiltered, "-*r")
    plt.plot(X, Ystar_mean, "--b")
    plt.show()
    
