import math
import numpy as np
import numerical.numpytheano as nt
import numerical.numpyext.linalg as ntl


def RBF_kernel(X1=np.eye(2), X2=np.eye(2), L=1, S=1, ns=nt.NumpyLinalg):
    return ns.exp(L) * ns.exp(-0.5*ntl.sqeuclidean(X1, X2) / ns.exp(S))

def noise_kernel(X1=np.eye(2), X2=np.eye(2), N=1, ns=nt.NumpyLinalg):
    """
    Uniform Gaussian noise kernel
    :param X1:
    :param X2:
    :param N: log of noise variance
    :param ns: operations namespace
    :return: Gram matrix
    """
    if X1 is X2:
        return ns.exp(N) * ns.identity(X1.shape[0])
    else:
        return ns.zeros([X1.shape[0], X2.shape[0]])

def RBF_2order_kernel(X1=np.eye(2), X2=np.eye(2), L=1, S1=1, S2=1, ns=nt.NumpyLinalg):
    if X1 is X2:
        X1 = Xtminus(X1, 2)
        X2 = X1
    else:
        X1 = Xtminus(X1, 2)
        X2 = Xtminus(X2, 2)
    return ns.exp(L) * ns.exp(-0.5*ntl.sqeuclidean(X1[0], X2[0]) / ns.exp(S1)
                              -0.5*ntl.sqeuclidean(X1[1], X2[1]) / ns.exp(S2))

def noise_2order_kernel(X1=np.eye(2), X2=np.eye(2), N=1, ns=nt.NumpyLinalg):
    """
    Uniform Gaussian noise kernel
    :param X1:
    :param X2:
    :param N: log of noise variance
    :param ns: operations namespace
    :return: Gram matrix
    """
    if X1 is X2:
        return ns.exp(N) * ns.identity(X1.shape[0]-2)
    else:
        return ns.zeros([X1.shape[0]-2, X2.shape[0]-2])

def RBF_2order_multi_sequence_kernel(X1=np.eye(3), X2=np.eye(3), X1tplusIndexes=[2], X2tplusIndexes=[2],
                                     L=1, S1=1, S2=1, ns=nt.NumpyLinalg):
    return ns.exp(L) * ns.exp(-0.5*ntl.sqeuclidean(X1[X1tplusIndexes-1], X2[X2tplusIndexes-1]) / ns.exp(S1)
                              -0.5*ntl.sqeuclidean(X1[X1tplusIndexes-2], X2[X2tplusIndexes-2]) / ns.exp(S2))

def noise_2order_multi_sequence_kernel(X1=np.eye(3), X2=np.eye(3), X1tplusIndexes=[2], X2tplusIndexes=[2],
                                       N=1, ns=nt.NumpyLinalg):
    """
    Uniform Gaussian noise kernel
    :param X1:
    :param X2:
    :param N: log of noise variance
    :param ns: operations namespace
    :return: Gram matrix
    """
    if X1 is X2:
        return ns.exp(N) * ns.identity(X1tplusIndexes.shape[0])
    else:
        return ns.zeros([X1tplusIndexes.shape[0], X2tplusIndexes.shape[0]])

def uniform_gauss_loglik(Q=2, X=np.eye(2), sigmasqr=1.0):
    return -0.5 * Q * X.shape[0] * math.log(2*math.pi) \
           -0.5 * Q * math.log(1) \
           -0.5 * (X * X).sum() / sigmasqr

def gplvm_loglik(Q=2, N=2, kernel=np.eye(2), Y=np.eye(2), ns=nt.NumpyLinalg):
    """
    GPLVM dimensionality reduction log-likelihood
    :param Q:
    :param N:
    :param kernel:
    :param Y:
    :param ns:
    :return:
    """
    return -0.5 * Q * N * math.log(2*math.pi) \
           -0.5 * Q * ns.log(ns.det(kernel)) \
           -0.5 * ns.trace(ns.dot(ns.choleskyinv(kernel), ns.dot(Y, Y.T)))

def Xtminus(X, order):
    """
    Constructs dynamics kernel arguments list
    :param X:
    :param order: 1 for 1-st order, 2 for 2-nd order...
    :return:
    """
    return [X[0+i:-order+i, :] for i in range(order)]

def Xtplus(X, order):
    """
    Constructs dynamics mapping output values
    :param X:
    :param order:
    :return:
    """
    return X[order:, :]

def conditional_gaussian(X, Ymean, Ycentered, Kinv, Kfunction, xStar, fullCov=False, order=0):
        xStar = xStar.copy()
        KxxStar = Kfunction(X, xStar)
        KxxStarKInv = np.dot(KxxStar.T, Kinv)
        yStar = Ymean + np.dot(KxxStarKInv, Ycentered)
        if fullCov:
            covStar = Kfunction(xStar, xStar) - np.dot(KxxStarKInv, KxxStar)
        else:
            if not isinstance(xStar, list):
                xS = [xStar[i:i+order+1, :] for i in range(xStar.shape[0]-order)]
                covStar = np.hstack([Kfunction(x, x) for x in xS]).ravel() \
                          - (KxxStarKInv * KxxStar.T).sum(axis=1)
            else:
                #  TODO: implement for the multidimensional input
                # xS = [xStar[i:i+order+1, :] for i in range(xStar.shape[0]-order)]
                # covStar = np.hstack([Kfunction(x, x) for x in xS]).ravel() \
                #           - (KxxStarKInv * KxxStar.T).sum(axis=1)
                covStar = None
        return yStar, covStar



def gpdm_loglik(Q=2, N=2, kernelXtminus=np.eye(2), Xtplus=np.eye(2), ns=nt.NumpyLinalg):
    """
    Dynamics only log-lokelihood for Xtminus->Xtplus GP mapping
    :param Q:
    :param N:
    :param kernelXtminus:
    :param Xtplus:
    :param ns:
    :return:
    """
    return -0.5 * Q * N * math.log(2*math.pi) \
           -0.5 * Q * ns.log(ns.det(kernelXtminus)) \
           -0.5 * ns.trace(ns.dot(ns.choleskyinv(kernelXtminus), ns.dot(Xtplus, Xtplus.T)))

def gplvm_gpdm_loglik(Q=2, N=2, dynamicsOrder=1, kernelLVM=np.eye(2), kernelDM=np.eye(2), X=np.eye(2), Y=np.eye(2), ns=nt.NumpyLinalg):
    """
    GPLVM + GPDM log-likelihood
    :param Q:
    :param N:
    :param kernelLVM:
    :param kernelDM:
    :param X:
    :param Y:
    :param ns:
    :return:
    """
    gplvm_part = gplvm_loglik(Q, N, kernelLVM, Y, ns)
    gpdm_part = gpdm_loglik(Q, N, kernelDM, Xtplus(X, dynamicsOrder), ns)
    x0_part = uniform_gauss_loglik(Q, X[0, :])
    if dynamicsOrder == 2:
        x0_part += uniform_gauss_loglik(Q, X[1, :]-X[0, :])
    return gplvm_part + gpdm_part + x0_part

def gplvm_gpdm_multi_sequence_loglik(Q=2, N=2, dynamicsOrder=1, kernelLVM=np.eye(2), kernelDM=np.eye(2),
                                     X=np.eye(2), Y=np.eye(2), Xt0indexes=[0], ns=nt.NumpyLinalg):
    """
    GPLVM + GPDM log-likelihood
    :param Q:
    :param N:
    :param kernelLVM:
    :param kernelDM:
    :param X:
    :param Y:
    :param ns:
    :return:
    """
    gplvm_part = gplvm_loglik(Q, N, kernelLVM, Y, ns)
    gpdm_part = gpdm_loglik(Q, N, kernelDM, Xtplus(X, dynamicsOrder), ns)
    x0_part = uniform_gauss_loglik(Q, X[Xt0indexes, :])
    if dynamicsOrder == 2:
        x0_part += uniform_gauss_loglik(Q, X[Xt0indexes+1, :]-X[Xt0indexes, :])
    return gplvm_part + gpdm_part + x0_part

#######################################################
#  Generic multi-dimensional tensor kernels
#######################################################

def noise_kern(X1=[np.eye(3)], X2=None, logNoiseVariance=1, ns=nt.NumpyLinalg):
    """
    Uniform Gaussian noise kernel for D-dimensional points
    :param X1: list of T data tensors, NxD dimensions each. X1[t][data_point_index, dim].
    :param X2: list of T data tensors, NxD dimensions each, or None if X1 == X2. X1[t][data_point_index, dim].
    :param logNoiseVariance: log of noise variance
    :param ns: operations namespace
    :return: Gram matrix
    """
    # Wrap into lists
    if X2 is None:
        X2 = X1
    X1isX2 = X1 is X2
    if not isinstance(X1, list):
        X1 = [X1]
    if not isinstance(X2, list):
        X2 = [X2]
    # Check dimensions agree
    assert len(X1) == len(X2)
    if ns == nt.NumpyLinalg:
        assert X1[0].shape[1:] == X1[0].shape[1:]

    if X1isX2:
        return ns.exp(logNoiseVariance) * ns.identity(X1[0].shape[0])
    else:
        return ns.zeros([X1[0].shape[0], X2[0].shape[0]])

def RBF_kern(X1=[np.eye(3)], X2=None, logScale=1, logLength=[1], ns=nt.NumpyLinalg):
    """
    RBF kernel for D-dimensional points
    :param X1: list of T data tensors, NxD dimensions each. X1[t][data_point_index, dim].
    :param X2: list of T data tensors, NxD dimensions each, or None if X1 == X2. X1[t][data_point_index, dim].
    :param logScale: log-Scale
    :param logLength: python list of T log-lengths for each of T data tensors
    :param ns: operations namespace
    :return: Gram matrix
    """
    # Wrap into lists
    if X2 is None:
        X2 = X1
    if not isinstance(X1, list):
        X1 = [X1]
    if not isinstance(X2, list):
        X2 = [X2]
    if not isinstance(logLength, list):
        logLength = [logLength]
    # Check dimensions agree
    assert len(X1) == len(logLength)
    assert len(X2) == len(logLength)
    if ns == nt.NumpyLinalg:
        assert X1[0].shape[1:] == X1[0].shape[1:]

    res = ns.zeros([X1[0].shape[0], X2[0].shape[0]])
    for i in range(len(logLength)):
        res -= 0.5*ntl.sqeuclidean(X1[i], X2[i]) / ns.exp(logLength[i])
    res = ns.exp(logScale + res)  # ns.exp(logScale) * ns.exp(res)
    return res

def linear_kern(X1=[np.eye(3)], X2=None, logScale=[1], ns=nt.NumpyLinalg):
    """
    Linear kernel for D-dimensional points
    :param X1: list of T data tensors, NxD dimensions each. X1[t][data_point_index, dim].
    :param X2: list of T data tensors, NxD dimensions each, or None if X1 == X2. X1[t][data_point_index, dim].
    :param logScale: python list of T log-lengths for each of T data tensors
    :param ns: operations namespace
    :return: Gram matrix
    """
    # Wrap into lists
    if X2 is None:
        X2 = X1
    if not isinstance(X1, list):
        X1 = [X1]
    if not isinstance(X2, list):
        X2 = [X2]
    if not isinstance(logScale, list):
        logScale = [logScale]
    # Check dimensions agree
    assert len(X1) == len(logScale)
    if ns == nt.NumpyLinalg:
        assert X1[0].shape[1:] == X1[0].shape[1:]

    res = ns.zeros([X1[0].shape[0], X2[0].shape[0]])
    for i in range(len(logScale)):
        res += ns.exp(logScale[i]) * X1[i].dot(X2[i].T)
    return res


def gplvm_loglik_2(Q=2, kernel=np.eye(2), Y=np.eye(2), ns=nt.NumpyLinalg):
    """
    GPLVM dimensionality reduction log-likelihood
    :param Q:
    :param kernel:
    :param Y:
    :param ns:
    :return:
    """
    N = kernel.shape[0]
    return -0.5 * Q * N * math.log(2*math.pi) \
           -0.5 * Q * ns.log(ns.det(kernel)) \
           -0.5 * ns.trace(ns.dot(ns.choleskyinv(kernel), ns.dot(Y, Y.T)))

def gpdm_loglik_2(Q=2, kernelXtminus=np.eye(2), Xtplus=np.eye(2), ns=nt.NumpyLinalg):
    """
    Dynamics only log-lokelihood for Xtminus->Xtplus GP mapping
    :param Q:
    :param N:
    :param kernelXtminus:
    :param Xtplus:
    :param ns:
    :return:
    """
    N = kernelXtminus.shape[0]
    return -0.5 * Q * N * math.log(2*math.pi) \
           -0.5 * Q * ns.log(ns.det(kernelXtminus)) \
           -0.5 * ns.trace(ns.dot(ns.choleskyinv(kernelXtminus), ns.dot(Xtplus, Xtplus.T)))
           # Alternative log-det, numerically more stable, much slower
           # -ns.log(ns.diag(ns.cholesky(kernelXtminus))).sum() \


def sequences_indexes(Ysequences=[[1, 2, 3], [4, 5, 6]]):
        sequencesends = np.cumsum([Yseq.shape[0] for Yseq in Ysequences])
        sequencesstarts = np.cumsum([0] + [Yseq.shape[0] for Yseq in Ysequences])[:-1]
        sequencesindexes = [np.arange(sequencesstarts[i], sequencesends[i]) for i in range(len(Ysequences))]
        return sequencesindexes

def xt0_xtminus_xtplus_indexes(sequencesindexes=[[1, 2, 3], [4, 5, 6]], dynamicsOrder=1):
    xt0indexes = np.array([si[0] for si in sequencesindexes])
    xtminusindexes = np.hstack([np.array(si[:-dynamicsOrder]) for si in sequencesindexes])
    xtminusindexes = [i + xtminusindexes for i in range(dynamicsOrder)]
    xtplusindexes = np.hstack([np.array(si[dynamicsOrder:]) for si in sequencesindexes])
    return xt0indexes, xtminusindexes, xtplusindexes

def xtminus_lists(X=[[1], [2], [3], [4], [5]], xtminusindexes=[[1, 2, 3], [2, 3, 4]]):
    """
    Constructs dynamics kernel arguments list
    :param X:
    :param xtminusindexes: 1 for 1-st order, 2 for 2-nd order...
    :return:
    """
    return [X[xind] for xind in xtminusindexes]

def gplvm_gpdm_loglik_2(Q=2, dynamicsOrder=1, kernelLVM=np.eye(2), kernelDM=np.eye(2),
                        X=np.eye(2), Y=np.eye(2),
                        xt0indexes=np.array([0, 20, 30]),
                        xtplusindexes=np.array([3, 4, 5, 6]),
                        ns=nt.NumpyLinalg):
    gplvm_part = gplvm_loglik_2(Q, kernelLVM, Y, ns)
    gpdm_part = gpdm_loglik_2(Q, kernelDM, X[xtplusindexes], ns)
    x0_part = uniform_gauss_loglik(Q, X[xt0indexes, :])
    if dynamicsOrder == 2:
        x0_part += uniform_gauss_loglik(Q, X[xt0indexes+1, :]-X[xt0indexes, :])
    return gplvm_part + gpdm_part + x0_part

def couple_kerns(partial_kerns, log_sigma_sqrs, X1_is_X2=True, ns=nt.NumpyLinalg):
    """
    Creates a list coupling kernels
    :param partial_kerns: list of M partial kernels, all are NxN matrices
    :param log_sigma_sqrs: MxM matrix of logs of sigma^2_{i,j}
    :param X1_is_X2: indicates whether arguments for the kernels were equal; kron(X1, X2)
    :param ns: operations namespace
    :return: list of M coupled kernels
    """
    sigmas_sqr = ns.exp(log_sigma_sqrs)
    sigmas_r_sqr = 1.0/((1.0/sigmas_sqr).sum(axis=1))
    n_kerns = len(partial_kerns)
    kerns_shape = partial_kerns[0].shape
    coupled_kerns = []
    for i in range(n_kerns):
        coupled_kern = ns.zeros(kerns_shape)
        for j in range(n_kerns):
            coupled_kern += (sigmas_r_sqr[i]**2 / sigmas_sqr[j, i]**2) * partial_kerns[j]
        if X1_is_X2:
            coupled_kern += sigmas_r_sqr[i] * ns.identity(kerns_shape[0])
        coupled_kerns.append(coupled_kern)
    return coupled_kerns

def gplvm_cgpdm_loglik(Qs=[2, 2], dynamics_order=1,
                       LVM_kernels=[np.eye(2), np.eye(2)],
                       DM_kernels=[np.eye(2), np.eye(2)],
                       Xs=[np.eye(2), np.eye(2)],
                       Ys=[np.eye(2), np.eye(2)],
                       xt0indexes=np.array([0, 20, 30]),
                       xtplusindexes=np.array([3, 4, 5, 6]),
                       partsindexes=[np.array([0, 1, 2]), np.array([3, 4, 5, 6])],
                       ns=nt.NumpyLinalg):
    gplvm_part = 0
    gpdm_part = 0
    x0_part = 0
    for i in range(len(partsindexes)):
        gplvm_part += gplvm_loglik_2(Qs[i], LVM_kernels[i], Ys[i], ns)
        gpdm_part += gpdm_loglik_2(Qs[i], DM_kernels[i], Xs[i][xtplusindexes], ns)
        x0_part += uniform_gauss_loglik(Qs[i], Xs[i][xt0indexes, :])
        if dynamics_order == 2:
            x0_part += uniform_gauss_loglik(Qs[i], Xs[i][xt0indexes+1, :]-Xs[i][xt0indexes, :])
    return gplvm_part + gpdm_part + x0_part




















