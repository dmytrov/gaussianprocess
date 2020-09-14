import numpy as np
import matplotlib.pyplot as plt
import matplotlibex as plx
import numerical.numpyext.linalg as ntlinalg


def gauss_pdf(xs, mean, covar):
    if len(xs.shape) == 1:
        xs = xs[np.newaxis, :]
    precision = np.linalg.inv(covar)
    z = np.sqrt(np.linalg.det(2*np.pi*covar))
    e = np.exp(-0.5 * np.sum((xs-mean) * precision.dot(xs.T-mean).T, axis=1))
    return 1.0/z * e
    
def product_of_gaussians(means, covars):
    M = len(means)
    assert len(means) == len(covars)
    precisions = [np.linalg.inv(c) for c in covars]
    precision = np.sum(np.stack(precisions), axis=0)
    covar = np.linalg.inv(precision)
    precisionsmeans = [p.dot(m) for p, m in zip(precisions, means)]
    precisionsmeanssum = np.sum(np.stack(precisionsmeans), axis=0)
    mean = covar.dot(precisionsmeanssum)
    
    k = np.prod([gauss_pdf(m[np.newaxis, :], 0.0, c) for m, c in zip(means, covars)]) / \
            gauss_pdf(mean[np.newaxis, :], 0.0, covar)
    scale = 1.0 / k
    return mean, covar, scale


if __name__ == "__main__":
    np.random.seed(6)
    N = 100
    X = np.linspace(-5, 5, N)[:, np.newaxis]
    mean = np.array([1.0])
    covar = np.array([[1.0]])
    g = gauss_pdf(X, mean, covar)
    print(np.sum(g)*(X[1]-X[0]))
    
    fig = plt.figure(figsize=(5, 5))
    plt.plot(X, g)
    plt.show()
    
    
    means = [np.array([-1.0]), np.array([1.0])]
    covars = [np.array([[1.0]]), np.array([[1.0]])]
    m, c, s = product_of_gaussians(means, covars)
    print(m, c, s)
    a = gauss_pdf(means[0] - means[1], 0, covars[0]+covars[1])
    print(1.0/a)
    
    fig = plt.figure(figsize=(5, 5))
    g0 = gauss_pdf(X, means[0], covars[0])
    g1 = gauss_pdf(X, means[1], covars[1])
    plt.plot(X, g0)
    plt.plot(X, g1)
    plt.plot(X, g0*g1)
    plt.plot(X, g0*g1*s, "*")
    plt.plot(X, gauss_pdf(X, m, c))
    plt.show()
    
    
