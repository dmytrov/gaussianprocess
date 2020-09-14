import time
import numpy as np

def matrix_inverse():
    mu = 0.0
    sigma = 1.0
    n = 2000
    x_vec = np.random.normal(mu, sigma, n**2)
    x = np.reshape(x_vec, (n, n))
    xinv = np.linalg.inv(x)
    sqerr = np.sum((x.dot(xinv) - np.identity(n))**2)
    print("Error: {}".format(sqerr))

def matrix_svd():
    mu = 0.0
    sigma = 1.0
    n = 3000
    d = 100
    x_vec = np.random.normal(mu, sigma, n*d)
    x = np.reshape(x_vec, (n, d))
    S, v, D = np.linalg.svd(x)
    V = np.zeros((n, d))
    V[:d, :d] = np.diag(v)
    sqerr = np.sum((x - np.dot(S, np.dot(V, D)))**2)
    print("Error: {}".format(sqerr))

def scalar_divide():
    mu = 0.0
    sigma = 1.0
    n = 10000
    x_vec = np.random.normal(mu, sigma, n**2)
    x = np.reshape(x_vec, (n, n))
    a = 0.9 * x[0, :]
    for i in range(10):
        x = x / a

def scalar_mult():
    mu = 0.0
    sigma = 1.0
    n = 10000
    x_vec = np.random.normal(mu, sigma, n**2)
    x = np.reshape(x_vec, (n, n))
    a = 0.9 * x[0, :]
    for i in range(10):
        x = x * a

def test_performance(func):
    t0 = time.time()
    print("Running {}".format(func.__name__))
    func()
    t1 = time.time()
    print("Time elapsed: {}".format(t1-t0))


if __name__ == "__main__":
    np.show_config()
    test_performance(matrix_inverse)
    test_performance(matrix_svd)
    test_performance(scalar_divide)
    test_performance(scalar_mult)
    
