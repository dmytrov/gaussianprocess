from __future__ import print_function
import numpy as np
import numpy.linalg.linalg as linalg

SHOW_PLOT_ON_SINGULAR_MATRIX = False

def report_singular_matrix(x):
    if SHOW_PLOT_ON_SINGULAR_MATRIX:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(x)
        plt.title("Singular matrix encountered")
        plt.show()


def cholesky_inv_jitter(x, maxiter=10):
    """
    Computes inverse of a positive (semi)definite matrix.
    Uses Cholesky decomposition + jitter if x is not positive definite.
    NumPy/Theano compatible
    :param x: a positive (semi)definite matrix
    :return: interse
    """
    try:
        L = np.linalg.cholesky(x)
        Linv = np.linalg.inv(L)
        return np.dot(Linv.T, Linv)
    except linalg.LinAlgError as err:
        if err.args[0] == "Matrix is not positive definite" or \
           err.args[0] == "Singular matrix":
            jitter = 1e-6 * abs(np.mean(np.diag(x)))
            print("Matrix shape: {}. cholesky_inv_jitter.".format(x.shape), end="")
            report_singular_matrix(x)
            for i in range(maxiter):
                try:
                    L = np.linalg.cholesky(x + jitter * np.eye(x.shape[0]))
                    Linv = np.linalg.inv(L)
                    print("OK")
                    return np.dot(Linv.T, Linv)
                except linalg.LinAlgError as err:
                    if err.args[0] == "Matrix is not positive definite" or \
                       err.args[0] == "Singular matrix":
                        jitter *= 10
                        print(".", end="")
        print("FAILED")
        print("Matrix: ", x) 
        print("Jitter: ", jitter) 
        raise


def cholesky_jitter(x, maxiter=10):
    """
    Computes Cholesky decomposition of a positive (semi)definite matrix.
    Uses Cholesky decomposition + jitter if x is not positive definite.
    NumPy/Theano compatible
    :param x: a positive (semi)definite matrix
    :return: interse
    """
    try:
        L = np.linalg.cholesky(x)
        return L
    except linalg.LinAlgError as err:
        if err.args[0] == "Matrix is not positive definite" or \
           err.args[0] == "Singular matrix":
            jitter = 1e-6 * abs(np.mean(np.diag(x)))
            print("Matrix shape: {}. cholesky_jitter.".format(x.shape), end="")
            report_singular_matrix(x)
            for i in range(maxiter):
                try:
                    L = np.linalg.cholesky(x + jitter * np.eye(x.shape[0]))
                    print("OK")
                    return L
                except linalg.LinAlgError as err:
                    if err.args[0] == "Matrix is not positive definite" or \
                       err.args[0] == "Singular matrix":
                        jitter *= 10
                        print(".", end="")
        print("FAILED")
        print("Matrix: ", x) 
        print("Jitter: ", jitter) 
        raise


def L2normsquared(A=np.eye(2)):
    '''
    L2 squared norms of the vectors matrix A
    NumPy/Theano compatible
    :param A: matrix [NxD] of N vectors
    :return: vector of N L2 squared norms
    '''
    return (A * A).sum(axis=1)

def sqeuclidean(A=np.eye(2), B=np.eye(2)):
    """
    Computes squared euclidean distance matrix of two sets of datapoints dinemsion D
    NumPy/Theano compatible
    :param A: [NxD]
    :param B: [MxD]
    :return: [NxM] of squared euclidean distances
    """
    ABT = A.dot(B.T)
    AATdiag = (A * A).sum(axis=1)[:, np.newaxis]
    BBTdiag = (B * B).sum(axis=1)[:, np.newaxis]
    res = (AATdiag + BBTdiag.T) - 2 * ABT
    return res

def quadratic_form(A, measure, B=None):
    """
    Quadratic form
    :param A: [NxD], N points of dimensionality D
    :param measure: [DxD] Symmetric Positive (semi)Definite matrix
    :param B: [MxD], M points of dimensionality D
    :return: [NxM] (A[n,:]-B[m,:])^T * measure * (A[n,:]-B[m,:]) 
             for all A[n,:], B[m,:], n in [1..N], m in [1..M]
             or [N] if B is none
    """
    # N, D = A.shape
    # M, _ = B.shape
    if B is None:
        measure_A = measure.dot(A.T)
        return np.sum(A.T * measure_A, axis=0)
    else:
        A_sub_B = A[:, np.newaxis, :] - B[np.newaxis, :, :]
        measure_A_sub_B = np.dot(A, measure)[:, np.newaxis, :] - np.dot(B, measure)[np.newaxis, :, :]
        return np.sum(A_sub_B * measure_A_sub_B, axis=2)


if __name__ == "__main__":
    # Construct a low-rank matrix xxt
    x = np.random.normal(size=[4, 2])
    xxt = x.dot(x.T)
    print(xxt)
    
    xxt_inv = cholesky_inv_jitter(xxt)
    print(xxt_inv)
    print(xxt.dot(xxt_inv))
    