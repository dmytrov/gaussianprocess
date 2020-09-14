import numpy as np
import numerical.numpytheano as nt
import numerical.numpyext.linalg as ntla
import numerical.numpyext as npx
import theano
import theano.tensor as tt
import matplotlibex as plx


def gaussian_z(covar, ns=nt.NumpyLinalg):
    if (hasattr(covar, "shape") and not hasattr(covar, "ndim")) or (hasattr(covar, "ndim") and covar.ndim==2):
        return ns.sqrt((2*ns.pi)**covar.shape[-1] * ns.det(covar))
    else:
        return ns.sqrt((2*ns.pi) * covar)

def gaussians_zs(covars, ns=nt.NumpyLinalg):
    return ns.sqrt((2*ns.pi)**covars.shape[-1] * ns.det(covars))

def log_gaussian_z(covar, ns=nt.NumpyLinalg):
    if (hasattr(covar, "shape") and not hasattr(covar, "ndim")) or (hasattr(covar, "ndim") and covar.ndim==2):
        return 0.5 * covar.shape[-1] * ns.log(2*ns.pi) + 0.5 * ns.log_det(covar)
    else:
        return 0.5 * ns.log(2*ns.pi) + 0.5 * ns.log(covar)

def log_gaussians_zs(covars, ns=nt.NumpyLinalg):
    return 0.5 * covars.shape[-1] * ns.log(2*ns.pi) + 0.5 * ns.log_det(covars)

def gaussian_isotropic_z(covar, ndims, ns=nt.NumpyLinalg):
    return ns.sqrt((2*ns.pi*covar)**ndims)

def log_gaussian_isotropic_z(covar, ndims, ns=nt.NumpyLinalg):
    return 0.5*ndims*ns.log(2*ns.pi*covar)

def gaussian_pdf(x, mean, covar, ns=nt.NumpyLinalg):
    return 1.0 / gaussian_z(covar, ns) * exp(-0.5 * ns.sum((x-mean) * ns.dot(ns.inv(covar), (x-mean))))

def log_gaussian_pdf(x, mean, covar, ns=nt.NumpyLinalg):
    return -log_gaussian_z(covar, ns) -0.5 * ns.sum((x-mean) * ns.dot(ns.inv(covar), (x-mean)))

def log_gaussians_pdfs(xs, means, covars, ns=nt.NumpyLinalg):
    return -log_gaussians_zs(covars, ns) -0.5 * ns.sum((xs-means) * ns.sum(ns.inv(covars) * (xs-means)[:, ns.newaxis, :], axis=-1), axis=-1)


def gaussian_entropy(covar, ns=nt.NumpyLinalg):
    """
    Calculates entropy of a Gaussian distribution
    :param covar: [D*D] covariance matrix
    :return: entropy
    """
    D = covar.shape[-1]
    return 0.5 * D * (1 + log(2*ns.pi)) + 0.5 * ns.det(covar)

def gaussians_entropy(covars, ns=nt.NumpyLinalg):
    """
    Calculates entropy of an array Gaussian distributions
    :param covar: [N*D*D] covariance matrices
    :return: total entropy
    """
    N = covar.shape[0]
    D = covar.shape[-1]
    return 0.5 * N * D * (1 + log(2*ns.pi)) + 0.5 * ns.sum(ns.det(covar))

def gaussians_diags_entropy(covars, ns=nt.NumpyLinalg):
    """
    Calculates entropy of an array Gaussian distributions each with diagonal covariance
    :param covar: [N*D] diagonal covariance vectors
    :return: total entropy
    """
    N = covars.shape[0]
    D = covars.shape[-1]
    return 0.5 * N * D * (1 + ns.log(2*ns.pi)) + 0.5 * ns.sum(ns.log(covars))

def gaussians_cross_entropy(mean1, covar1, mean2, covar2, ns=nt.NumpyLinalg):
    """
    Calculates cross-entropy of two Gaussian distributions H(p_1, p_2) = \int p_1 log(p_2) dx
    :param mean1: [D] mean_1
    :param covar1: [D*D] covariance matrix_1
    :param mean2: [D] mean_2
    :param covar2: [D*D] covariance matrix_2
    :return: cross-entropy
    """
    D = covar1.shape[-1]
    return -0.5 * ns.trace(ns.inv(covar2).dot(covar1)) + log_gaussian_pdf(mean1, mean2, covar2, ns)

def gaussians_cross_entropies(means1, covars1, means2, covars2, ns=nt.NumpyLinalg):
    """
    Calculates cross-entropies of N pairs of Gaussian distributions H(p_1, p_2) = \int p_1 log(p_2) dx
    :param means1: [N*D] means_1
    :param covars1: [N*D*D] covariance matrices_1
    :param means2: [N*D] means_2
    :param covars2: [N*D*D] covariance matrices_2
    :return: [N] cross-entropies
    """
    D = covars1.shape[-1]
    return -0.5 * ns.trace(ns.sum(ns.inv(covars2)[:, ns.newaxis, :, :] * covars1[:, :, :, ns.newaxis], axis=-2), axis1=-1, axis2=-2) \
           + log_gaussians_pdfs(means1, means2, covars2, ns)

def optimal_q_u(kzzinv, noise_covar, x_mean, psi_1, psi_2, ns=nt.NumpyLinalg):
    """
    Computes optimal q(u)
    :param kzzinv: [M*M]
    :param noise_covar: scalar
    :param x_mean: [N*D] means of GP mappinf outpot x_t: GP(x_minus_t) -> f_t -> x_t
    :param psi_1: [N*M]
    :param psi_2: [N*M*M]
    :return: u_mean [M*D], u_covar [M*M]
    """
    natural_param_1 = ns.dot(kzzinv, ns.dot(psi_1.T, x_mean)) / noise_covar  # [M*D]
    psi_2_sum = ns.sum(psi_2, axis=0)
    natural_param_2 = -0.5 * (kzzinv +  ns.dot(kzzinv, ns.dot(psi_2_sum, kzzinv)) / noise_covar)  # [M*M]
    u_covar = ns.inv(-2.0 * natural_param_2)
    u_mean = ns.dot(u_covar, natural_param_1)
    return u_mean, u_covar


class RBF_ARD_kern(object):
    def __init__(self):
        pass

    @classmethod
    def gram_matrix(cls, sigmasqrf, lambdas, x1, x2, ns=nt.NumpyLinalg):
        """
        ARD RBF kernel (Gram) matrix
        :param sigmasqrf: scalar
        :param lambdassqr: [D] vector
        :param x1: [NxD] matrix
        :param x2: [MxD] matrix
        :return: [NxM] Gram matrix
        """
        lambdasqrtinv = 1.0 / ns.sqrt(lambdas[ns.newaxis, :])
        x1scaled = lambdasqrtinv * x1
        x2scaled = lambdasqrtinv * x2
        return sigmasqrf * ns.exp(-0.5 * ntla.sqeuclidean(x1scaled, x2scaled))

    @classmethod
    def psi_stat_0(cls, sigmasqrf, x_means, ns=nt.NumpyLinalg):
        """
        Psi_0 kernel statistics
        :param sigmasqrf: scalar
        :param x_means: [NxD]
        :return: [N]
        """
        return ns.ones_like(x_means[:, 0]) * sigmasqrf

    @classmethod
    def psi_stat_1(cls, sigmasqrf, lambdas, aug_z, x_means, x_covars, ns=nt.NumpyLinalg):
        """
        Psi_1 kernel statistics
        :param sigmasqrf: scalar
        :param lambdas: [D]
        :param aug_z: [MxD]
        :param x_means: [NxD]
        :param x_covars: [NxDxD]
        :return: [NxM]
        """
        x_means_sub_aug_z = x_means[:, ns.newaxis, :] - aug_z[ns.newaxis, :, :]  # [N*M*D]
        lambda_mat = ns.diag(lambdas)[ns.newaxis, :, :]  # [1*D*D] broadcastable to [N*D*D]
        measures = ns.inv(lambda_mat + x_covars)  # [N*D*D]
        # Matrix prod
        r1 = ns.sum(measures[:, ns.newaxis, :, :] * x_means_sub_aug_z[:, :, ns.newaxis, :], axis=-1)  # [N*M*D]
        r2 = ns.sum(x_means_sub_aug_z * r1, axis=2)  # [N*M]
        r3 = ns.exp(-0.5 * r2)  # [N*M]
        r4 = ns.prod(lambdas)  # scalar
        r5 = ns.det(lambda_mat + x_covars)  # [N]
        r6 = sigmasqrf * ns.sqrt(r4 / r5)[:, ns.newaxis]  # [N*1]
        res = r6 * r3  # [N*M]
        return  res

    @classmethod
    def psi_stat_2(cls, sigmasqrf, lambdas, aug_z, x_means, x_covars, ns=nt.NumpyLinalg):
        """
        Psi_2 kernel statistics
        :param sigmasqrf: scalar
        :param lambdas: [D]
        :param aug_z: [MxD]
        :param x_means: [NxD]
        :param x_covars: [NxDxD]
        :return: [NxMxM]
        """ 
        D = x_means.shape[-1]

        r0 = ns.diag(0.5 * lambdas)[ns.newaxis, :, :] + x_covars  # [N*D*D]
        r1 = ns.inv(r0)  # [N*D*D]
        r2 = 0.5 * (aug_z[ns.newaxis, :, :] - aug_z[:, ns.newaxis, :])  # [M*M*D]
        # The Gaussian is [N*M*M]
        r1 = r1[:, ns.newaxis, ns.newaxis, :, :]  # [N*M*M*D*D]
        r2 = r2[ns.newaxis, :, :, :]  # [N*M*M*D]
        r3 = x_means[:, ns.newaxis, ns.newaxis, :]  # [N*M*M*D]
        r4 = r2-r3  # [N*M*M*D]
        # Matrix prod
        r5 = ns.sum(r1 * r4[:, :, :, ns.newaxis,:], axis=-1)  # [N*M*M*D]
        # Matrix prod
        r6 = ns.sum(r5 * r4, axis=3)  # [N*M*M]
        r7 = 1 / ns.sqrt((2*ns.pi)**D * ns.det(r0))[:, ns.newaxis, ns.newaxis]  # [N*M*M]
        r8 = r7 * ns.exp(-0.5*r6)  # [N*M*M] - the rightmost Gaussians

        c1 = aug_z[:, ns.newaxis, :] - aug_z[ns.newaxis, :, :]  # [M*M*D]
        #c2 = ns.diag(2.0 * lambdas)  # [D*D]
        c3 = c1 / (2.0 * lambdas)[ns.newaxis, ns.newaxis, :]  # [M*M*D]
        #c3 = np.tensordot(c1, c2, [[2], [1]])  # [M*M*D]
        #c4 = ns.inner(c1, c3)  # [M*M]
        c4 = ns.sum(c1 * c3, axis=2)  # [M*M]
        c5 = c4[ns.newaxis, :, :]  # [N*M*M]
        c6 = 1.0 / ns.sqrt((2*ns.pi)**D * 2.0**D * ns.prod(lambdas))  # scalar
        c7 = c6 * ns.exp(-0.5*c4)  # [M*M] - the central Gaussians

        l1 = sigmasqrf**2 * (2*ns.pi)**D * ns.prod(lambdas)

        res = l1 * c7 * r8
        return res


class RBF_ARD_kern_diag_x_covar(object):
    def __init__(self):
        pass

    @classmethod
    def gram_matrix(cls, sigmasqrf, lambdas, x1, x2, ns=nt.NumpyLinalg):
        """
        ARD RBF kernel (Gram) matrix
        :param sigmasqrf: scalar
        :param lambdassqr: [D] vector
        :param x1: [NxD] matrix
        :param x2: [MxD] matrix
        :return: [NxM] Gram matrix
        """
        lambdasqrtinv = 1.0 / ns.sqrt(lambdas[ns.newaxis, :])
        x1scaled = lambdasqrtinv * x1
        x2scaled = lambdasqrtinv * x2
        gram = sigmasqrf * ns.exp(-0.5 * ntla.sqeuclidean(x1scaled, x2scaled))
        #boost = 1e-6 * ns.mean(ns.diag(gram)) * ns.ones_like(gram)
        return gram #+ boost

    @classmethod
    def psi_stat_0(cls, sigmasqrf, x_means, ns=nt.NumpyLinalg):
        """
        Psi_0 kernel statistics
        :param sigmasqrf: scalar
        :param x_means: [NxD]
        :return: [N]
        """
        return ns.ones_like(x_means[:, 0]) * sigmasqrf

    @classmethod
    def psi_stat_1(cls, sigmasqrf, lambdas, aug_z, x_means, x_covars, ns=nt.NumpyLinalg):
        """
        Psi_1 kernel statistics
        :param sigmasqrf: scalar
        :param lambdas: [D]
        :param aug_z: [MxD]
        :param x_means: [NxD]
        :param x_covars: [NxD]
        :return: [NxM]
        """
        x_means_sub_aug_z = x_means[:, ns.newaxis, :] - aug_z[ns.newaxis, :, :]  # [N*M*D]
        lambda_mat = lambdas[ns.newaxis, :]  # [1*D] broadcastable to [N*D]
        measures = 1.0 / (lambda_mat + x_covars)  # [N*D]
        measures = measures[:, ns.newaxis, :]  # [N*1*D] broadcastable to [N*M*D]
        # Matrix prod
        r2 = ns.sum(x_means_sub_aug_z * measures * x_means_sub_aug_z, axis=2)  # [N*M]
        r3 = ns.exp(-0.5 * r2)  # [N*M]
        r4 = ns.prod(lambdas)  # scalar
        r5 = ns.prod(lambda_mat + x_covars, axis=1)  # [N]
        r6 = sigmasqrf * ns.sqrt(r4 / r5)[:, ns.newaxis]  # [N*1]
        res = r6 * r3  # [N*M]
        return  res

    @classmethod
    def psi_stat_1_Lawrence(cls, sigmasqrf, lambdas, aug_z, x_means, x_covars):
        """
        Psi_1 kernel statistics
        :param sigmasqrf: scalar
        :param lambdas: [D]
        :param aug_z: [MxD]
        :param x_means: [NxD]
        :param x_covars: [NxD]
        :return: [NxM]
        """
        Q = x_means.shape[-1]
        mu_n = x_means[0]
        S_n = x_covars[0]
        alpha = 1.0 / lambdas
        z = aug_z
        psi1 = sigmasqrf
        for q in range(Q):
            num = np.exp(-0.5 * (alpha[q] * (mu_n[q] - z[:, q])**2) / (alpha[q] * S_n[q] + 1.0))
            den = np.sqrt(alpha[q] * S_n[q] + 1)
            psi1 *= num / den
        return psi1

        

    @classmethod
    def psi_stat_2(cls, sigmasqrf, lambdas, aug_z, x_means, x_covars, ns=nt.NumpyLinalg):
        """
        Psi_2 kernel statistics
        :param sigmasqrf: scalar
        :param lambdas: [D]
        :param aug_z: [MxD]
        :param x_means: [NxD]
        :param x_covars: [NxD]
        :return: [NxMxM]
        """ 
        D = x_means.shape[-1]

        r0 = 0.5 * lambdas[ns.newaxis, :] + x_covars  # [N*D]
        r1 = 1.0 / r0  # [N*D]
        r2 = 0.5 * (aug_z[ns.newaxis, :, :] + aug_z[:, ns.newaxis, :])  # [M*M*D]
        # The Gaussian is [N*M*M]
        r1 = r1[:, ns.newaxis, ns.newaxis, :]  # [N*M*M*D]
        r2 = r2[ns.newaxis, :, :, :]  # [N*M*M*D]
        r3 = x_means[:, ns.newaxis, ns.newaxis, :]  # [N*M*M*D]
        r4 = r2-r3  # [N*M*M*D]
        # Matrix prod
        r6 = ns.sum(r4 * r1 * r4, axis=3)  # [N*M*M]
        r7 = 1 / ns.sqrt((2*ns.pi)**D * ns.prod(r0, axis=1))[:, ns.newaxis, ns.newaxis]  # [N*M*M]
        r8 = r7 * ns.exp(-0.5*r6)  # [N*M*M] - the rightmost Gaussians

        c1 = aug_z[ns.newaxis, :, :] - aug_z[:, ns.newaxis, :]  # [M*M*D]
        #c2 = ns.diag(2.0 * lambdas)  # [D*D]
        c3 = 1.0 / (2.0 * lambdas)[ns.newaxis, ns.newaxis, :]  # [M*M*D]
        #c3 = np.tensordot(c1, c2, [[2], [1]])  # [M*M*D]
        #c4 = ns.inner(c1, c3)  # [M*M]
        c4 = ns.sum(c1 * c3 * c1, axis=2)  # [M*M]
        c5 = c4[ns.newaxis, :, :]  # [N*M*M]
        c6 = 1.0 / ns.sqrt((2*ns.pi)**D * 2.0**D * ns.prod(lambdas))  # scalar
        c7 = c6 * ns.exp(-0.5*c4)  # [M*M] - the central Gaussians

        l1 = sigmasqrf**2 * (2*ns.pi)**D * ns.prod(lambdas)

        res = l1 * c7 * r8
        return res

    @classmethod
    def psi_stat_2_Lawrence(cls, sigmasqrf, lambdas, aug_z, x_means, x_covars):
        """
        Psi_2 kernel statistics
        :param sigmasqrf: scalar
        :param lambdas: [D]
        :param aug_z: [MxD]
        :param x_means: [NxD]
        :param x_covars: [NxD]
        :return: [NxMxM]
        """ 
        Q = x_means.shape[-1]
        mu_n = x_means[0]
        S_n = x_covars[0]
        alpha = 1.0 / lambdas
        z = aug_z
        psi2 = sigmasqrf**2
        z_bar = 0.5 * (z[np.newaxis, :, :] + z[:, np.newaxis, :])
        for q in range(Q):
            num = np.exp(-(alpha[q] * (z[np.newaxis, :, q] - z[:, np.newaxis, q])**2) / 4.0 
                         -(alpha[q] * (mu_n[q] - z_bar[:, :, q])**2) / (2.0 * alpha[q] * S_n[q] + 1.0))
            den = np.sqrt(2 * alpha[q] * S_n[q] + 1)
            psi2 *= num / den
        return psi2

class VariationalGPLVM(object):
    """
    Variational GPLVM.
    Mapping of N points: X\in[N*Q] -> Y\in[N*D]
    Augmented by M mappings: Z\in[M*Q] -> U\in[M*D]
    Reference paper: Tistial and Lawrence (2010) - Bayesian GPLVM.
    """

    @classmethod
    def elbo(cls, y, beta, Kzz, psi_stat_0, psi_stat_1, psi_stat_2, ns=nt.NumpyLinalg):
        """
        :param y: [N*D], observed points, centered
        :param beta: scalar, X->Y mapping noise
        :param Kzz: [M*M], augmenting points covariance
        :param psi_stat_0: [N]
        :param psi_stat_1: [N*M]
        :param psi_stat_2: [N*M*M]
        :return: scalar, ELBO
        """
        N, D = y.shape
        psi_stat_0_sum = ns.sum(psi_stat_0)
        psi_stat_2_sum = ns.sum(psi_stat_2, axis=0)  # [M*M]
        W = beta * ns.identity(N) - beta**2 * psi_stat_1.dot(ns.inv(beta * psi_stat_2_sum + Kzz).dot(psi_stat_1.T))  # [N*N]
        res = 0.5 * D * N * ns.log(beta) \
            + 0.5 * D * ns.log(ns.det(Kzz)) \
            - 0.5 * D * N * ns.log(2*ns.pi) \
            - 0.5 * D * ns.log(ns.det(beta * psi_stat_2_sum + Kzz)) \
            - 0.5 * ns.sum(y * ns.dot(W, y)) \
            - 0.5 * D * beta * psi_stat_0_sum \
            + 0.5 * D * beta * ns.trace(ns.inv(Kzz).dot(psi_stat_2_sum))
        return res

class VariationalGPDM(object):
    """
    Variational GPDM (state-space).
    nminus - dynamics order
    Mapping of N points: X_minust\in[N*(D*nminus)] -> X_t\in[N*D]
    Augmented by M mappings: Z\in[M*D] -> U\in[M*D]
    """

    @classmethod
    def elbo_kron(cls, xt_means, xt_covars, alpha, Kzz, psi_stat_0, psi_stat_1, psi_stat_2, ns=nt.NumpyLinalg):
        """
        Computes the ELBO with Kronecker matrix expansions
        :param xt_means: [N*Q], variational latent means, x_t, mapping outputs
        :param xt_covars: [N*Q*Q], variational latent covariances for x_t, mapping outputs
        :param alpha: scalar, X->X mapping noise
        :param Kzz: [M*M], augmenting points covariance
        :param psi_stat_0: [N]
        :param psi_stat_1: [N*M]
        :param psi_stat_2: [N*M*M]
        :return: scalar, ELBO
        """
        N, Q = xt_means.shape
        N, M = psi_stat_1.shape
        
        psi_stat_0_sum = ns.sum(psi_stat_0)  # scalar
        psi_stat_2_sum = ns.sum(psi_stat_2, axis=0)  # [M*M]
        Kzzinv = ns.inv(Kzz)  # [M*M]
        
        A = ns.identity(Q)  # Kronecker product for the kernel function
        Kzzkron = ns.kron(Kzz, A)
        Kzzinvkron = ns.kron(Kzzinv, A)
        psi_stat_1_kron = ns.kron(psi_stat_1[:, ns.newaxis, :], A)  # [N*Q*MQ]
        #psi_stat_1_kron = ns.kron(psi_stat_1, ns.array([1]*Q))  # [N*MQ]
        psi_stat_2_kron = ns.kron(psi_stat_2, A)  # [N*MQ*MQ]
        psi_stat_2_kron_sum = ns.sum(psi_stat_2, axis=0)  # [MQ*MQ]
        alphakron = alpha * A

        FF = 1.0/alpha * Kzzinvkron.dot(ns.sum(psi_stat_2_kron, axis=0).dot(Kzzinvkron))
        FFinv = ns.inv(FF)
        s = ns.sum(xt_means[:, :, ns.newaxis] * psi_stat_1_kron, axis=(0, 1))
        #s = 0
        #s = ns.zeros([1, M*Q])
        #for t in range(N):
        #    s += ns.dot(xt_means[t], psi_stat_1_kron[t])
        GG = 1.0/alpha * s.dot(Kzzinvkron).T
        HH = - N * ns.log(gaussian_z(alphakron, ns=ns)) \
	        -0.5 / alpha * ns.sum(xt_means * xt_means) \
	        -0.5 / alpha * ns.sum(xt_covars) \
            -0.5 / alpha * ns.sum(ns.trace(psi_stat_0[:, ns.newaxis, ns.newaxis] - ns.sum(Kzzinvkron[ns.newaxis, :, :, ns.newaxis] * psi_stat_2_kron[:, ns.newaxis, :, :], axis=-2), axis1=1, axis2=2))  # -0.5 / alpha * Q * (M * psi_stat_0_sum - ns.trace(ns.dot(Kzzinv, psi_stat_2_sum)))
        #HH = 0
        #for t in range(N):
        #    HH += -np.log(gaussian_z(alphakron)) \
        #        -0.5 * xt_means[t].dot(xt_means[t]) \
        #        -0.5 * np.trace(xt_covars[t] / alpha) \
        #        -0.5 * np.trace(psi_stat_0[t]/alpha - Kzzinvkron.dot(psi_stat_2_kron[t]) / alpha)
        II = -log_gaussian_z(Kzzkron + FFinv, ns) \
            -0.5 * GG.T.dot(FFinv.dot(ns.inv((FFinv + Kzzinvkron)).dot(FFinv.dot(GG)))) \
            +ns.log(gaussian_z(FFinv, ns)) \
            +0.5 * GG.T.dot(FFinv.dot(GG)) \
            +HH
        return II

    @classmethod
    def elbo_diag_x_covars(cls, xt_means, xt_covars, alpha, Kzz, psi_stat_0, psi_stat_1, psi_stat_2, ns=nt.NumpyLinalg):
        """
        Computes the ELBO with Kronecker matrix expansions
        :param xt_means: [N*Q], variational latent means, x_t, mapping outputs
        :param xt_covars: [N*Q], variational latent covariances for x_t, mapping outputs
        :param alpha: scalar, X->X mapping noise
        :param Kzz: [M*M], augmenting points covariance
        :param psi_stat_0: [N]
        :param psi_stat_1: [N*M]
        :param psi_stat_2: [N*M*M]
        :return: scalar, ELBO
        """
        N, Q = xt_means.shape
        N, M = psi_stat_1.shape
        
        psi_stat_0_sum = ns.sum(psi_stat_0)  # scalar
        psi_stat_2_sum = ns.sum(psi_stat_2, axis=0)  # [M*M]

        Kzzinv = ns.inv(Kzz)  # [M*M]
        FFinv = alpha * Kzz.dot(ns.inv(psi_stat_2_sum).dot(Kzz)) 
        #FFinv = alpha * Kzz.dot(ns.choleskyinv(psi_stat_2_sum).dot(Kzz))             
        #FF = 1.0 / alpha * Kzzinv.dot(psi_stat_2_sum.dot(Kzzinv))
        #FFinv = ns.inv(FF)
        GG = 1.0 / alpha * ns.dot(xt_means.T, psi_stat_1).dot(Kzzinv).T
        HH = - N * Q * log_gaussian_z(alpha, ns=ns) \
	        -0.5 / alpha * ns.sum(xt_means * xt_means) \
	        -0.5 / alpha * ns.sum(xt_covars) \
            -0.5 / alpha * Q * (psi_stat_0_sum - ns.trace(ns.dot(Kzzinv, psi_stat_2_sum)))
        II = -Q * log_gaussian_z(Kzz + FFinv, ns) \
            -0.5 * ns.trace(GG.T.dot(FFinv.dot(ns.inv((FFinv + Kzz)).dot(FFinv.dot(GG))))) \
            +Q * log_gaussian_z(FFinv, ns) \
            +0.5 * ns.trace(GG.T.dot(FFinv.dot(GG))) \
            +HH
        xt_covars_entropy = gaussians_diags_entropy(xt_covars, ns=ns)
        return II + xt_covars_entropy


if __name__ == "__main__":
    print("===============================")
    X = np.array([[0, 1], [1, 2], [5, 5]])
    means = np.array([[0, 0], [1, 1]])
    measure = np.array([[1, 0], [0, 1]])
    print(ntla.quadratic_form(X, measure, means))

    print("===============================")
    A = np.array([[1, 2], [3, 4]])
    k = 10
    result = 1
    for i in range(k):
        result = result * A
    print(result)
    res_np = result

    result = npx.scan(
        fn=lambda prior_result, A: prior_result * A,
        outputs_info=np.ones_like(A),
        non_sequences=A,
        n_steps=k)
    res_scan = result[-1]
    print(res_scan)

    assert np.all(res_np == res_scan)

    print("===============================")
    X = np.array([[0, 1], [1, 2], [-1, 3], [2, 2]])
    N, D = X.shape
    x_means = X - np.mean(X, axis=0)
    x_covars = np.array([[[1, 0], [0, 1]]] * N)
    aug_z = np.array([[1, -1], [3, 2], [-1, 1]])
    RBF_ARD_kern(x_means, x_covars, aug_z)
    sigmasqrf = 1
    lambdas = np.array([2.0, 2.0])
    k = RBF_ARD_kern.gram_matrix(sigmasqrf, lambdas, x_means, x_means)
    print(k)

    psi1 = RBF_ARD_kern.psi_stat_1(sigmasqrf, lambdas, aug_z, x_means, x_covars)
    print(psi1)
    psi1_ = RBF_ARD_kern.psi_stat_1_(sigmasqrf, lambdas, aug_z, x_means, x_covars)
    print(psi1_)
    psi2 = RBF_ARD_kern.psi_stat_2(sigmasqrf, lambdas, aug_z, x_means, x_covars)
    print(psi2)


