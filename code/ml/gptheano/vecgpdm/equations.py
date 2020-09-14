import numpy as np
import numerical.numpytheano as nt
import numerical.numpyext.linalg as ntla
import matplotlibex as plx
import matplotlib.pyplot as plt



def list_of_nones(N):
    return [None for i in range(N)]

def matrix_of_nones(N, M):
    return [list_of_nones(M) for i in range(N)]


def gaussian_z(covar, ns=nt.NumpyLinalg):
    if (hasattr(covar, "shape") and not hasattr(covar, "ndim")) or \
            (hasattr(covar, "ndim") and covar.ndim==2):
        return ns.sqrt((2*ns.pi)**covar.shape[-1] * ns.det(covar))
    else:
        return ns.sqrt((2*ns.pi) * covar)

def gaussians_zs(covars, ns=nt.NumpyLinalg):
    return ns.sqrt((2*ns.pi)**covars.shape[-1] * ns.det(covars))

def log_gaussian_z(covar, ns=nt.NumpyLinalg):
    if (hasattr(covar, "shape") and not hasattr(covar, "ndim")) or \
            (hasattr(covar, "ndim") and covar.ndim==2):
        return 0.5 * (covar.shape[-1] * ns.log(2*ns.pi) + ns.log_det(covar))
    else:
        return 0.5 * (ns.log(2*ns.pi) + ns.log(covar))

def log_gaussians_zs(covars, ns=nt.NumpyLinalg):
    return 0.5 * (covars.shape[-1] * ns.log(2*ns.pi) + ns.log_det(covars))

def gaussian_isotropic_z(covar, ndims, ns=nt.NumpyLinalg):
    return ns.sqrt((2*ns.pi*covar)**ndims)

def log_gaussian_isotropic_z(covar, ndims, ns=nt.NumpyLinalg):
    return 0.5*ndims*ns.log(2*ns.pi*covar)

def gaussian_pdf(x, mean, covar, ns=nt.NumpyLinalg):
    return 1.0 / gaussian_z(covar, ns) \
            * ns.exp(-0.5 * ns.sum((x-mean) * ns.dot(ns.inv(covar), (x-mean))))

def log_gaussian_pdf(x, mean, covar, ns=nt.NumpyLinalg):
    return -log_gaussian_z(covar, ns) \
            -0.5 * ns.sum((x-mean) * ns.dot(ns.inv(covar), (x-mean)))

def log_gaussians_pdfs(xs, means, covars, ns=nt.NumpyLinalg):
    return -log_gaussians_zs(covars, ns) \
            -0.5 * ns.sum((xs-means) \
            * ns.sum(ns.inv(covars) * (xs-means)[:, ns.newaxis, :], axis=-1), \
            axis=-1)


def gaussian_entropy(covar, ns=nt.NumpyLinalg):
    """
    Calculates entropy of a Gaussian distribution
    :param covar: [D*D] covariance matrix
    :return: entropy
    """
    D = covar.shape[-1]
    return 0.5 * D * (1 + ns.log(2*ns.pi)) + 0.5 * ns.det(covar)

def gaussians_entropy(covars, ns=nt.NumpyLinalg):
    """
    Calculates entropy of an array Gaussian distributions
    :param covars: [N*D*D] covariance matrices
    :return: total entropy
    """
    N = covars.shape[0]
    D = covars.shape[-1]
    return 0.5 * N * D * (1 + ns.log(2*ns.pi)) + 0.5 * ns.sum(ns.log_det(covars))

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
    #D = covar1.shape[-1]
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
    #D = covars1.shape[-1]
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

def optimal_q_u_2(kzzinv, noise_covar, x_mean, psi_stat_1, psi_stat_2, ns=nt.NumpyLinalg):
    """
    Computes optimal q(u)
    :param kzzinv: [M*M]
    :param noise_covar: scalar
    :param x_mean: [N*D] means of GP mappinf outpot x_t: GP(x_minus_t) -> f_t -> x_t
    :param psi_1: [N*M]
    :param psi_2: [N*M*M]
    :return: u_mean [M*D], u_covar [M*M]
    """
    psi_stat_2_sum = ns.sum(psi_stat_2, axis=0)  # [M*M]
    FF = 1.0 / noise_covar * kzzinv.dot(psi_stat_2_sum.dot(kzzinv))
    GG = 1.0 / noise_covar * ns.dot(x_mean.T, psi_stat_1).dot(kzzinv).T
    u_covar = ns.inv(kzzinv + FF)
    u_mean = ns.dot(u_covar, GG)
    return u_mean, u_covar

def gaussian_PoE(means, covars, ns=nt.NumpyLinalg):
    """
    Product of Gaussian experts. Computes the resulting Gaussian mean and covariance
    :param means: [N, D] - N mean vectors of dimensionality D
    :param covars: [N, D, D] or [N, D] or [N] - N covariance matrices or isotropic diagonal variances
    :return: mean [D], covariance [D, D]
    """
    N = means.shape[0]
    if covars.ndim == 3:
        precisions = ns.inv(covars)
        poe_precision = ns.sum(precisions, axis=0)
        poe_covar = ns.inv(poe_precision)
        poe_mean_sum = 0
        for i in range(N):
            poe_mean_sum += ns.dot(precisions[i, :, :], means[i, :])
        poe_mean = poe_covar.dot(poe_mean_sum)
    elif covars.ndim == 2:
        precisions = 1.0 / covars
        poe_precision = ns.sum(precisions, axis=0)
        poe_covar = 1.0 / poe_precision
        poe_mean_sum = 0
        for i in range(N):
            poe_mean_sum += precisions[i, :] * means[i, :]
        poe_mean = poe_covar * poe_mean_sum
    elif covars.ndim == 1:
        if means.ndim == 2:
            means = means[np.newaxis, :, :]
        N = means.shape[1]
        precisions = 1.0 / covars
        poe_precision = ns.sum(precisions, axis=0)
        poe_covar = 1.0 / poe_precision
        poe_mean_sum = 0
        for i in range(N):
            poe_mean_sum += precisions[i] * means[:, i, :]
        poe_mean = poe_covar * poe_mean_sum
    else: 
        raise ValueError("covars argument is invalid")
    return poe_mean, poe_covar
    
    

class RBF_ARD_Kern(object):
    """
    RBF ARD kernel for full covariance x_covars matrix.
    Computationally more expensive than the diagonal covariance case
    """
    def __init__(self):
        pass

    @classmethod
    def gram_matrix(cls, sigmasqrf, lambdas, x1, x2, ns=nt.NumpyLinalg):
        """
        ARD RBF kernel (Gram) matrix
        :param sigmasqrf: scalar
        :param lambdas: [D] vector
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
        #c5 = c4[ns.newaxis, :, :]  # [N*M*M]
        c6 = 1.0 / ns.sqrt((2*ns.pi)**D * 2.0**D * ns.prod(lambdas))  # scalar
        c7 = c6 * ns.exp(-0.5*c4)  # [M*M] - the central Gaussians

        l1 = sigmasqrf**2 * (2*ns.pi)**D * ns.prod(lambdas)

        res = l1 * c7 * r8
        return res


class RBF_ARD_Kern_diag_x_covar(object):
    """
    ARD RBF kernel for diagonal covariance x_covars case
    """

    @classmethod
    def gram_matrix(cls, sigmasqrf, lambdas, x1, x2, ns=nt.NumpyLinalg):
        """
        ARD RBF kernel (Gram) matrix
        :param sigmasqrf: scalar
        :param lambdas: [D] vector
        :param x1: [NxD] matrix
        :param x2: [MxD] matrix
        :return: [NxM] Gram matrix
        """
        lambdasqrtinv = 1.0 / ns.sqrt(lambdas[ns.newaxis, :])
        x1scaled = lambdasqrtinv * x1
        x2scaled = lambdasqrtinv * x2
        gram = sigmasqrf * ns.exp(-0.5 * ntla.sqeuclidean(x1scaled, x2scaled))
        return gram

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
        #c5 = c4[ns.newaxis, :, :]  # [N*M*M]
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


class Linear_ARD_Kern_diag_x_covar(object):
    """
    ARD RBF kernel for diagonal covariance x_covars case
    """

    @classmethod
    def gram_matrix(cls, lambdas, x1, x2, ns=nt.NumpyLinalg):
        """
        ARD RBF kernel (Gram) matrix
        :param lambdas: [D] vector
        :param x1: [NxD] matrix
        :param x2: [MxD] matrix
        :return: [NxM] Gram matrix
        """
        lambdasinv = 1.0 / lambdas
        x1_lambda = lambdasinv[ns.newaxis, :] * x1
        gram = ns.sum(x1_lambda[:, np.newaxis, :] * x2[np.newaxis, :, :], axis=2)
        return gram

    @classmethod
    def psi_stat_0(cls, lambdas, x_means, x_covars, ns=nt.NumpyLinalg):
        """
        Psi_0 kernel statistics
        :param lambdas: [D] vectors
        :param x_means: [NxD]
        :param x_covars: [NxD]
        :return: [N]
        """
        lambdasinv = 1.0 / lambdas
        res = ns.sum(lambdasinv[np.newaxis, :] * (x_means**2 + x_covars), axis=1)
        return res

    @classmethod
    def psi_stat_1(cls, lambdas, aug_z, x_means, ns=nt.NumpyLinalg):
        """
        Psi_1 kernel statistics
        :param lambdas: [D]
        :param aug_z: [MxD]
        :param x_means: [NxD]
        :return: [NxM]
        """
        lambdasinv = 1.0 / lambdas
        x_means_lambda = lambdasinv[np.newaxis, :] * x_means
        res = ns.sum(x_means_lambda[:, np.newaxis, :] * aug_z[np.newaxis, :, :], axis=2)
        return  res

    @classmethod
    def psi_stat_2(cls, lambdas, aug_z, x_means, x_covars, ns=nt.NumpyLinalg):
        """
        Psi_2 kernel statistics
        :param lambdas: [D]
        :param aug_z: [MxD]
        :param x_means: [NxD]
        :param x_covars: [NxD]
        :return: [NxMxM]
        """
        lambdasinv = 1.0 / lambdas
        aug_z_lambda = lambdasinv[ns.newaxis, :] * aug_z  # [M*D]

        # Means part
        r1 = aug_z_lambda[ns.newaxis, :, ns.newaxis,:] * aug_z_lambda[:, ns.newaxis, :, ns.newaxis]  # [M*M*D*D]
        r2 = x_means[:, ns.newaxis, :] * x_means[:, :, ns.newaxis]  # [N*D*D]
        r3 = ns.sum(r1[np.newaxis, :, :, :, :] * r2[:, np.newaxis, np.newaxis, :, :], axis=(3, 4))  # [N*M*M]

        # Covars part
        r4 = aug_z_lambda[ns.newaxis, :, :] * aug_z_lambda[:, ns.newaxis, :]  # [M*M*D]
        r5 = ns.sum(r4[np.newaxis, :, :, :] * x_covars[:, np.newaxis, np.newaxis, :], axis=3)  # [N*M*M]
        res = r3 + r5
        return res


class RBF_plus_Linear_ARD_Kern_diag_x_covar(object):
    """
    ARD RBF + linear kernel for diagonal covariance x_covars case
    """

    @classmethod
    def gram_matrix(cls, sigmasqrf, lambdas_rbf, lambdas_linear, x1, x2, ns=nt.NumpyLinalg):
        """
        ARD RBF + linear kernel (Gram) matrix
        :param sigmasqrf: scalar
        :param lambdas_rbf: [D] vector
        :param lambdas_linear: [D] vector
        :param x1: [NxD] matrix
        :param x2: [MxD] matrix
        :return: [NxM] Gram matrix
        """
        gram_rbf = RBF_ARD_Kern_diag_x_covar.gram_matrix(sigmasqrf, lambdas_rbf, x1, x2, ns=ns)
        gram_lin = Linear_ARD_Kern_diag_x_covar.gram_matrix(lambdas_linear, x1, x2, ns=ns)
        return gram_rbf + gram_lin

    @classmethod
    def psi_stat_0(cls, sigmasqrf, lambdas_linear, x_means, x_covars, ns=nt.NumpyLinalg):
        """
        Psi_0 kernel statistics
        :param sigmasqrf: scalar
        :param lambdas_linear: [D] vectors
        :param x_means: [NxD]
        :param x_covars: [NxD]
        :return: [N]
        """
        psi_rbf = RBF_ARD_Kern_diag_x_covar.psi_stat_0(sigmasqrf, x_means, ns=ns)
        psi_lin = Linear_ARD_Kern_diag_x_covar.psi_stat_0(lambdas_linear, x_means, x_covars, ns=ns)
        return psi_rbf + psi_lin

    @classmethod
    def psi_stat_1(cls, sigmasqrf, lambdas_rbf, lambdas_linear, aug_z, x_means, x_covars, ns=nt.NumpyLinalg):
        """
        Psi_1 kernel statistics
        :param sigmasqrf: scalar
        :param lambdas_rbf: [D]
        :param lambdas_linear: [D] vectors
        :param aug_z: [MxD]
        :param x_means: [NxD]
        :param x_covars: [NxD]
        :return: [NxM]
        """
        psi_rbf = RBF_ARD_Kern_diag_x_covar.psi_stat_1(sigmasqrf, lambdas_rbf, aug_z, x_means, x_covars, ns=ns)
        psi_lin = Linear_ARD_Kern_diag_x_covar.psi_stat_1(lambdas_linear, aug_z, x_means, ns=ns)
        return psi_rbf + psi_lin

    @classmethod
    def psi2_lin_cross_rbf(cls, sigmasqrf, lambdas_rbf, lambdas_linear, aug_z, x_means, x_covars, ns=nt.NumpyLinalg):
        """
        Psi_2 cross-statistics between linear and RBF kernels
        :param sigmasqrf: scalar
        :param lambdas_rbf: [D]
        :param lambdas_linear: [D] vectors
        :param aug_z: [MxD]
        :param x_means: [NxD]
        :param x_covars: [NxD]
        :return: [NxMxM]
        """
        def gaussian_z(covar):
            return ns.sqrt((2.0 * np.pi)**covar.shape[-1] \
                    * ns.prod(covar, axis=-1))
            

        def gaussian_pdf(x, mean, covar):
            return 1.0 / gaussian_z(covar) \
                    * ns.exp(-0.5*ns.sum(((x-mean)**2)/covar, axis=-1))

        aug_z0 = aug_z[:, np.newaxis, :]  # [NxMxMxD]
        aug_z1 = aug_z[np.newaxis, :, :]  # [NxMxMxD]
        A = aug_z0 / lambdas_linear  # [NxMxMxD]
        x_means = x_means[:, np.newaxis, np.newaxis, :]  # [NxMxMxD]
        x_covars = x_covars[:, np.newaxis, np.newaxis, :]  # [NxMxMxD]
        C = (aug_z1/lambdas_rbf + x_means/x_covars) \
                / (1.0/lambdas_rbf + 1.0/x_covars)  # [NxMxMxD]
        res = sigmasqrf * ns.sum(A * C, axis=-1) * gaussian_z(lambdas_rbf) \
                * gaussian_pdf(aug_z1, x_means, lambdas_rbf + x_covars)
        return res

    @classmethod
    def psi_stat_2(cls, sigmasqrf, lambdas_rbf, lambdas_linear, aug_z, x_means, x_covars, ns=nt.NumpyLinalg):
        """
        Psi_2 kernel statistics
        :param sigmasqrf: scalar
        :param lambdas_rbf: [D]
        :param lambdas_linear: [D] vectors
        :param aug_z: [MxD]
        :param x_means: [NxD]
        :param x_covars: [NxD]
        :return: [NxMxM]
        """
        psi_rbf = RBF_ARD_Kern_diag_x_covar.psi_stat_2(sigmasqrf, lambdas_rbf, aug_z, x_means, x_covars, ns=ns)  # [N*M*M]
        psi_lin = Linear_ARD_Kern_diag_x_covar.psi_stat_2(lambdas_linear, aug_z, x_means, x_covars, ns=ns)  # [N*M*M]
        psi_lin_cross_rbf = RBF_plus_Linear_ARD_Kern_diag_x_covar.psi2_lin_cross_rbf(sigmasqrf, lambdas_rbf, lambdas_linear, aug_z, x_means, x_covars, ns=ns)  # [N*M*M]
        psi_rbf_cross_lin = ns.transpose(psi_lin_cross_rbf, [0, 2, 1])  # [N*M*M]
        return psi_rbf + psi_lin + psi_lin_cross_rbf + psi_rbf_cross_lin  # [N*M*M]


class InvMode(object):
    simple = 0
    cholesky = 1
    

class GPLVM(object):
    inv_mode = InvMode.cholesky

    @classmethod
    def inv_op(cls, x, ns=nt.NumpyLinalg):
        if GPLVM.inv_mode == InvMode.simple:
            return ns.inv(x)
        elif GPLVM.inv_mode == InvMode.cholesky:
            return ns.choleskyinv(x)

    @classmethod
    def loglikelihood(cls, Kxx, y, beta, ns=nt.NumpyLinalg):
        """
        Log-likelihood of GPLVM
        :param Kxx: [N*N], kernel matrix
        :param y: [N*D], observed points, centered
        :param beta: scalar, X->Y mapping noise precision
        :return: scalar, log-likelihood
        """
        alpha = 0.0
        if beta is not None:
            alpha = 1.0 / beta
        N, D = y.shape
        Kxx_plus_noise = Kxx + ns.identity(N) * alpha
        res =  -0.5 * D * N * ns.log(2 * np.pi) \
               -0.5 * D * ns.log_det(Kxx_plus_noise) \
               -0.5 * ns.trace(GPLVM.inv_op(Kxx_plus_noise, ns).dot(y.dot(y.T)))
        return res

    @classmethod
    def F_optimal(cls, Kxx, y, beta, ns=nt.NumpyLinalg):
        """
        :param Kxx: [N*N], kernel matrix
        :param y: [N*D], observed points, centered
        :param beta: scalar, X_{-t}->X_{t} mapping noise precision
        :return: Foptimal - denoised y
        """
        N = Kxx.shape[0]
        Foptimal = (GPLVM.inv_op(GPLVM.inv_op(Kxx) + np.identity(N) * beta)).dot(y) * beta
        return Foptimal

    @classmethod
    def variational_elbo(cls, y, beta, Kzz, psi_stat_0, psi_stat_1, psi_stat_2, ns=nt.NumpyLinalg):
        """
        Variational GPLVM.
        Mapping of N points: X\in[N*Q] -> Y\in[N*D]
        Augmented by M mappings: Z\in[M*Q] -> U\in[M*D]
        Reference paper: Tistial and Lawrence (2010) - Bayesian GPLVM.
        :param y: [N*D], observed points, centered
        :param beta: scalar, X->Y mapping noise precision
        :param Kzz: [M*M], augmenting points covariance
        :param psi_stat_0: [N]
        :param psi_stat_1: [N*M]
        :param psi_stat_2: [N*M*M]
        :return: scalar, ELBO
        """
        N, D = y.shape
        psi_stat_0_sum = ns.sum(psi_stat_0)
        psi_stat_2_sum = ns.sum(psi_stat_2, axis=0)  # [M*M]
        W = beta * ns.identity(N) - beta**2 * psi_stat_1.dot(GPLVM.inv_op(beta * psi_stat_2_sum + Kzz, ns).dot(psi_stat_1.T))  # [N*N]
        res = 0.5 * D * N * ns.log(beta) \
            + 0.5 * D * ns.log_det(Kzz) \
            - 0.5 * D * N * ns.log(2*ns.pi) \
            - 0.5 * D * ns.log_det(beta * psi_stat_2_sum + Kzz) \
            - 0.5 * ns.sum(y * ns.dot(W, y)) \
            - 0.5 * D * beta * psi_stat_0_sum \
            + 0.5 * D * beta * ns.trace(GPLVM.inv_op(Kzz, ns).dot(psi_stat_2_sum))
        return res

class GPDM(object):
    inv_mode = InvMode.cholesky

    @classmethod
    def inv_op(cls, x, ns=nt.NumpyLinalg):
        if GPLVM.inv_mode == InvMode.simple:
            return ns.inv(x)
        elif GPLVM.inv_mode == InvMode.cholesky:
            return ns.choleskyinv(x)

    @classmethod
    def combined_PoE_uncertainty(cls, alpha, ns=nt.NumpyLinalg):
        """
        :param alpha: [M*M], [from, to] X_{-t}->X_{t} mapping noise variance matrix
        :return: [M] joint uncertainty
        """
        alpha_inv = 1.0 / alpha  # [W*W]
        alpha_joint_inv = ns.sum(alpha_inv, axis=0)  # [W]
        alpha_joint = 1.0 / alpha_joint_inv  # [W]
        return alpha_joint, alpha_joint_inv
    
    @classmethod
    def loglikelihood_single_x0(cls, X, order, ns=nt.NumpyLinalg):
        """
        :param X: [T*D], latent dynamics points
        :param order: integer, order of the dynamics
        :return: log p(x0)
        """
        N, D = X.shape
        res = 0
        if order >= 1:
            res += log_gaussian_pdf(X[0, :], 0.0 * X[0, :], ns.identity(D), ns)
        if order >= 2:
            res += log_gaussian_pdf(X[1, :], X[0, :], ns.identity(D), ns)
        return res
        
    @classmethod
    def loglikelihood_single(cls, Kxx, Xout, alpha=None, order=2, ns=nt.NumpyLinalg):
        """
        Log-likelihood of GPDM with single dynamics
        :param Kxx: [N*N], kernel matrix
        :param Xout: [T*D], latent dynamics points
        :param alpha: scalar, X_{-t}->X_{t} mapping noise variance
        :param order: integer, order of the dynamics
        :return: scalar, log-likelihood
        """
        N, D = Xout.shape
        if alpha is not None:
            Kxx += ns.identity(N) * alpha
        res =  -0.5 * D * N * ns.log(2 * np.pi) \
               -0.5 * D * ns.log_det(Kxx) \
               -0.5 * ns.trace(GPDM.inv_op(Kxx, ns).dot(Xout.dot(Xout.T)))
        return res

    @classmethod
    def loglikelihood_multiple(cls, Kxxs, Xouts, alphas=None, order=2, ns=nt.NumpyLinalg):
        """
        Log-likelihood of GPDM with multiple dynamices
        :param Kxx: W*[N*N], kernel matrix
        :param Xout: W*[T*D], latent dynamics points
        :param alpha: [W], X_{-t}->X_{t} mapping noise variance
        :param order: integer, order of the dynamics
        :return: scalar, log-likelihood
        """
        W = len(Kxxs)
        if alphas is None:
            alphas = [None] * W
        res = 0.0
        for Kxx, Xout, alpha in zip(Kxxs, Xouts, alphas):
            res += GPDM.loglikelihood_single(Kxx, Xout, alpha, order, ns=ns)
        return res

    @classmethod
    def loglikelihood_coupled_marginalized_Ks(cls, Kxxs, Xouts, alpha, order=2, ns=nt.NumpyLinalg):
        """
        Log-likelihood of CGPDM with marginalized out intermediate predictions
        and a list of M PoE kernels
        :param Kxxs: M*M*[N*N], array of kernel matrices for M parts
        :param Xouts: M*[N*D], list of latent dynamics output points
        :param alpha: [M*M], [from, to] X_{-t}->X_{t} mapping noise variance matrix
        :param order: integer, order of the dynamics
        :return: scalar, log-likelihood
        """
        res = 0.0
        M = len(Kxxs)
        alpha_joint, _ = GPDM.combined_PoE_uncertainty(alpha, ns)  # [W]
        Ks = []
        for i in range(M):
            Ki = 0.0
            for j in range(M):
                Ki += (alpha_joint[i] / alpha[j, i])**2 * Kxxs[j][i]
            Ks.append(Ki + ns.identity(Ki.shape[0]) * alpha_joint[i])
            res += GPDM.loglikelihood_single(Ki, Xouts[i], alpha_joint[i], order, ns=ns)
        return res, Ks

    @classmethod
    def loglikelihood_coupled_marginalized(cls, Kxxs, Xouts, alpha, order=2, ns=nt.NumpyLinalg):
        """
        Log-likelihood of CGPDM with marginalized out intermediate predictions
        :param Kxxs: M*M*[N*N], array of kernel matrices for M parts
        :param Xouts: M*[N*D], list of latent dynamics output points
        :param alpha: [M*M], [from, to] X_{-t}->X_{t} mapping noise variance matrix
        :param order: integer, order of the dynamics
        :return: scalar, log-likelihood
        """
        ll, ks = GPDM.loglikelihood_coupled_marginalized_Ks(Kxxs, Xouts, alpha, order=2, ns=ns)
        return ll


    @classmethod
    def loglikelihood_coupled_unmarginalised(cls, Kxxs, Xouts, alpha, order=2, ns=nt.NumpyLinalg):
        """
        Log-likelihood of GPDM with explicit unmarginalised couplings, the same as marginalised.
        :param Kxxs: M*M*[N*N], array of kernel matrices for M parts
        :param Xouts: M*[N*D], list of latent dynamics output points
        :param alpha: [M*M], [from, to] X_{-t}->X_{t} mapping noise variance matrix
        :param order: integer, order of the dynamics
        :return: scalar, log-likelihood
        """
        return GPDM.loglikelihood_coupled_marginalized(Kxxs, Xouts, alpha, order, ns=ns)

    @classmethod
    def F_optimal_single(cls, Kxx, Xout, alpha, order=2, ns=nt.NumpyLinalg):
        """
        :param Kxx: [N*N], kernel matrix
        :param Xout: [N*D], latent dynamics output points
        :param alpha: scalar, X_{-t}->X_{t} mapping noise variance
        :return: Foptimal - denoised Xout
        """
        N = Kxx.shape[0]
        beta = 1.0 / alpha
        Foptimal = (GPDM.inv_op(GPDM.inv_op(Kxx) + np.identity(N) * beta)).dot(Xout) * beta
        return Foptimal

    @classmethod
    def F_optimal_coupled_kron(cls, Kxxs, Xouts, alpha, order=2, ns=nt.NumpyLinalg):
        """
        Computes optimal F for unmarginalised CGPDM.
        See "MAP estimate of unmarginalised CGPDM" in the notes.
        :param Kxxs: M*M*[N*N], array of kernel matrices for M parts
        :param Xouts: M*[N*D], list of latent dynamics output points
        :param alpha: [M*M], X_{-t}->X_{t} mapping noise variance matrix
        :return: M*M*[N*D] optimal F
        """
        N = Kxxs[0][0].shape[0]
        M = len(Kxxs)
        Fopt = matrix_of_nones(M, M)
        alpha_joint, alpha_joint_inv = GPDM.combined_PoE_uncertainty(alpha, ns)  # [W]
        for i in range(M):
            Fi = GPDM.F_optimal_coupled_one_part([Kxxs[j][i] for j in range(M)], 
                                                 Xouts[i], 
                                                 Ai=ns.zeros([M, N]) + (alpha_joint[i] / alpha[:, i])[:, ns.newaxis],
                                                 Bi=ns.zeros([N]) + alpha_joint[i],
                                                 order=order,
                                                 ns=ns)
            for j in range(M):
                Fopt[j][i] = Fi[j]  # [N*D]
        return Fopt

    @classmethod
    def F_optimal_coupled_one_part(cls, Kxxs, Xouts, Ai, Bi, order=2, ns=nt.NumpyLinalg):
        """
        Computes optimal F for unmarginalised CGPDM from "couplings" vector, one part only.
        See "MAP estimate of unmarginalised CGPDM" in the notes.
        :param Kxxs: M*[N*N], array of kernel matrices for M parts
        :param Xouts: [N*D], list of latent dynamics output points
        :param Ai: [M*N], X_{-t}->X_{t} mapping kernels weights
        :param Bi: [N], X_{-t}->X_{t} mapping noise total variances
        :return: M*M*[N*D] optimal F
        """
        N = Kxxs[0][0].shape[0]
        M = len(Kxxs)
        Fopt = list_of_nones(M)
        Ai = ns.concatenate([ns.diag(Ai[j]) for j in range(M)] , axis=1)
        Biinv = ns.diag(1.0 / Bi) 
        # Construct the big block-diagonal matrix K
        Kis = matrix_of_nones(M, M)
        for j in range(M):
            for k in range(M):
                Kis[j][k] = ns.zeros([N, N])
            Kis[j][j] = Kxxs[j]
        Ki = ns.stack_mat(Kis)
        # Make sure Ki is full rank. 
        # It does not affect the solution but allows the Fopt equation to be applicable
        for i in range(Ki.shape[0]):
            if abs(Ki[i, i]) < 1.0e-6:
                Ki[i, i] = 1.0
        Kiinv = GPDM.inv_op(Ki)
        #plt.figure()
        #plt.imshow(Ki)
        #plt.imshow(Kiinv)
        # Compute F^{:,i}
        Fi = GPDM.inv_op(Ai.T.dot(Biinv).dot(Ai) + Kiinv).dot(Ai.T.dot(Biinv)).dot(Xouts)  # [NN*D]
        for j in range(M):
            Fopt[j] = Fi[N*j:N*(j+1), :]  # [N*D]
        return Fopt
        
    @classmethod
    def generate_mean_prediction_single(cls, gpf, x0, T, order=2):
        """
        Generates mean prediction trajectory for single GPDM
        :param gpf: GP autoregressive mapping function
        
        """
        X = np.array(x0)
        X = np.array(x0)
        if X.ndim < 2:
            X = X[np.newaxis, :]
        assert X.shape[0] == order
        for t in range(T-order):
            xt, _ = gpf.posterior_predictive(X[-order, :][:])
            X = np.vstack([X, xt])
        return X
        

    @classmethod
    def generate_mean_prediction_coupled(cls, gpfs, alpha, x0s, T, order=2):
        """
        Generates mean-prediction trajectory t=1..T for M parts
        :param gpfs[from][to]: (M)(M), kernel functions
        :param Kxxinvs[from][to]: (M)(M)[N, N], kernel matrices
        :param Fopts[from][to]: (M)(M)[N, D], optimal mapping functions outputs
        :param alpha[from, to]: [M, M], X_{-t}->X_{t} mapping noise variance matrix
        :x0s[m]: (M)[order, D_m] or (M)[L, order, D_m], start point or L start points
        :return: (M)[T, D_m], trajectories
        """
        M = len(x0s)
        x0s = [np.array(x0)[np.newaxis, :] if x0.ndim < 2 else np.array(x0) for x0 in x0s]  # [order, D_m]
        x0s = [np.array(x0)[np.newaxis, :, :] if x0.ndim < 3 else np.array(x0) for x0 in x0s]  # [L, order, D_m]
        L = x0s[0].shape[0]
        Xpath = []  # (M)[L, T, D_m]
        for i in range(M):
            Xpath.append(0.0 * np.ones((L, T, x0s[i].shape[-1])))
            Xpath[i][:, 0:order, :] = x0s[i]
        for t in range(order, T):
            Xt = [xm[:, t-order:t, :] for xm in Xpath]  # (M)[L, order, D_m]
            Xt = [np.reshape(xm, [L, -1]) for xm in Xt]
            for i in range(M):  # [j][i]. i - to, j - from
                pp = gpfs[0][i].posterior_predictive(Xt[0])[0]
                means = np.stack([gpfs[j][i].posterior_predictive(Xt[j])[0] for j in range(M)], axis=-2)
                covars = alpha[:, i]
                Xpath[i][:, t:t+order, :] = gaussian_PoE(means=means, covars=covars)[0][:, np.newaxis, :]
        if L == 1:
            Xpath = [np.squeeze(xp, axis=0) for xp in Xpath]
        return Xpath
        

    @classmethod
    def variational_elbo_kron(cls, xt_means, xt_covars, alpha, Kzz, psi_stat_0, psi_stat_1, psi_stat_2, ns=nt.NumpyLinalg):
        """
        Variational GPDM (state-space).
        nminus - dynamics order
        Mapping of N points: X_minust\in[N*(D*nminus)] -> X_t\in[N*D]
        Augmented by M mappings: Z\in[M*D] -> U\in[M*D]
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

        #psi_stat_0_sum = ns.sum(psi_stat_0)  # scalar
        #psi_stat_2_sum = ns.sum(psi_stat_2, axis=0)  # [M*M]
        Kzzinv = ns.inv(Kzz)  # [M*M]

        A = ns.identity(Q)  # Kronecker product for the kernel function
        Kzzkron = ns.kron(Kzz, A)
        Kzzinvkron = ns.kron(Kzzinv, A)
        psi_stat_1_kron = ns.kron(psi_stat_1[:, ns.newaxis, :], A)  # [N*Q*MQ]
        psi_stat_2_kron = ns.kron(psi_stat_2, A)  # [N*MQ*MQ]
        #psi_stat_2_kron_sum = ns.sum(psi_stat_2, axis=0)  # [MQ*MQ]
        alphakron = alpha * A

        FF = 1.0/alpha * Kzzinvkron.dot(ns.sum(psi_stat_2_kron, axis=0).dot(Kzzinvkron))
        FFinv = ns.inv(FF)
        s = ns.sum(xt_means[:, :, ns.newaxis] * psi_stat_1_kron, axis=(0, 1))
        GG = 1.0/alpha * s.dot(Kzzinvkron).T
        HH = - N * ns.log(gaussian_z(alphakron, ns=ns)) \
	        -0.5 / alpha * ns.sum(xt_means * xt_means) \
	        -0.5 / alpha * ns.sum(xt_covars) \
            -0.5 / alpha * ns.sum(ns.trace(psi_stat_0[:, ns.newaxis, ns.newaxis] - \
                                           ns.sum(Kzzinvkron[ns.newaxis, :, :, ns.newaxis] * psi_stat_2_kron[:, ns.newaxis, :, :], axis=-2), \
                                           axis1=1, axis2=2))  
        II = -log_gaussian_z(Kzzkron + FFinv, ns) \
            -0.5 * GG.T.dot(FFinv.dot(ns.inv((FFinv + Kzzinvkron)).dot(FFinv.dot(GG)))) \
            +ns.log(gaussian_z(FFinv, ns)) \
            +0.5 * GG.T.dot(FFinv.dot(GG)) \
            +HH
        return II

    @classmethod
    def variational_elbo_diag_x_covars(cls, xt_means, xt_covars, alpha, Kzz, psi_stat_0, psi_stat_1, psi_stat_2, ns=nt.NumpyLinalg):
        """
        Variational GPDM (state-space).
        nminus - dynamics order
        Mapping of N points: X_minust\in[N*(D*nminus)] -> X_t\in[N*D]
        Augmented by M mappings: Z\in[M*D] -> U\in[M*D]
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

        Kzzinv = GPDM.inv_op(Kzz, ns)  # [M*M]
        FFinv = alpha * Kzz.dot(GPDM.inv_op(psi_stat_2_sum, ns).dot(Kzz))
        #FFinv = alpha * Kzz.dot(ns.choleskyinv(psi_stat_2_sum).dot(Kzz))
        #FF = 1.0 / alpha * Kzzinv.dot(psi_stat_2_sum.dot(Kzzinv))
        #FFinv = ns.inv(FF)
        GG = 1.0 / alpha * ns.dot(xt_means.T, psi_stat_1).dot(Kzzinv).T
        HH = - N * Q * log_gaussian_z(alpha, ns=ns) \
	        -0.5 / alpha * ns.sum(xt_means * xt_means) \
	        -0.5 / alpha * ns.sum(xt_covars) \
            -0.5 / alpha * Q * (psi_stat_0_sum - ns.trace(ns.dot(Kzzinv, psi_stat_2_sum)))
        II = -Q * log_gaussian_z(Kzz + FFinv, ns) \
            -0.5 * ns.trace(GG.T.dot(FFinv.dot(GPDM.inv_op(FFinv + Kzz, ns).dot(FFinv.dot(GG))))) \
            +Q * log_gaussian_z(FFinv, ns) \
            +0.5 * ns.trace(GG.T.dot(FFinv.dot(GG))) \
            +HH
        xt_covars_entropy = gaussians_diags_entropy(xt_covars, ns=ns)
        return II + xt_covars_entropy

    @classmethod
    def variational_explicit_elbo_diag_x_covars(cls, xt_means, xt_covars, alpha, Kzz, psi_stat_0, psi_stat_1, psi_stat_2, ns=nt.NumpyLinalg):
        """
        Variational Coupled GPDM (state-space) with explicit (unintegrated) partial mappings.
        nminus - dynamics order
        Mapping of N points: X_minust\in[N*(D*nminus)] -> X_t\in[N*D]
        Augmented by M mappings: Z\in[M*D] -> U\in[M*D]
        Computes the ELBO for W coupled parts
        W - number of parts
        N - number of full dynamics iterations
        Q - number of dimensions in one dynamics
        M - number of augmenting points
        :param xt_means: W*[N*Q], list of W arrays of variational latent means, x_t, mapping outputs
        :param xt_covars: W*[N*Q], list of W arrays of variational latent covariances for x_t, mapping outputs
        :param alpha: [W*W], X->X mapping noise covariance
        :param Kzz: W*W*[M*M], augmenting points covariance
        :param psi_stat_0: W*W[N]
        :param psi_stat_1: W*W[N*M]
        :param psi_stat_2: W*W*[N*M*M]
        :return: scalar, ELBO
        """
        elbo = 0
        W = len(xt_means)
        alpha_inv = 1.0 / alpha
        alpha_joint, alpha_joint_inv = GPDM.combined_PoE_uncertainty(alpha, ns)  # [W]

        Kzzinv = matrix_of_nones(W, W)
        for i in range(W):
            for j in range(W):
                Kzzinv[j][i] = GPDM.inv_op(Kzz[j][i], ns)

        # Intermediate equations used for posterior predictive
        pp_FF_list = list_of_nones(W)  # W*[(WM)*(WM)]
        pp_GG_list = list_of_nones(W)  # W*[(WM)*Q]
        pp_Kzzdiag_list = list_of_nones(W)  # W*[(WM)*(WM)]

        for i in range(W):
            N, Q = xt_means[i].shape
            N, M = psi_stat_1[i][i].shape

            FF_i_list = matrix_of_nones(W, W)
            GG_i_list = list_of_nones(W)
            for j in range(W):
                GG_ijk = alpha_inv[j, i] * ns.dot(xt_means[i].T, psi_stat_1[j][i]).dot(Kzzinv[j][i])  # [Q*M]
                GG_i_list[j] = GG_ijk.T  # [M*Q]
                for k in range(W):
                    if j == k:
                        psi_stat_2_sum = ns.sum(psi_stat_2[j][i], axis=0)  # [M*M]
                    else:
                        psi_stat_2_sum = ns.sum(psi_stat_1[j][i][:, :, np.newaxis] * psi_stat_1[k][i][:, np.newaxis, :], axis=0)  # [M*M]
                    FF_ijk = alpha_joint[i] * alpha_inv[j, i] * alpha_inv[k, i] * Kzzinv[j][i].dot(psi_stat_2_sum.dot(Kzzinv[k][i]))  # [M*M]
                    FF_i_list[j][k] = FF_ijk
            GG_i = ns.concatenate(GG_i_list, axis=0)  # [(WM)*Q]
            FF_i = ns.stack_mat(FF_i_list)  # [(WM)*(WM)]
            FF_i_inv = GPDM.inv_op(FF_i, ns)  # [(WM)*(WM)]

            HH_i = 0
            for j in range(W):
                psi_stat_0_sum = ns.sum(psi_stat_0[j][i])  # scalar
                psi_stat_2_sum = ns.sum(psi_stat_2[j][i], axis=0)
                HH_i += -0.5 * alpha_joint[i] * alpha_inv[j, i]**2 * Q * ( psi_stat_0_sum  -  ns.trace(ns.dot(Kzzinv[j][i], psi_stat_2_sum)))
            HH_i += - N * Q * log_gaussian_z(alpha_joint[i], ns=ns)
            HH_i += -0.5 * alpha_joint_inv[i] * ns.sum(xt_means[i] * xt_means[i])
            HH_i += -0.5 * alpha_joint_inv[i] * ns.sum(xt_covars[i])


            Kzzdiag_i_list = matrix_of_nones(W, W)
            for j in range(W):
                for k in range(W):
                    Kzzdiag_i_list[j][k] = ns.zeros_like(Kzz[j][k])
                Kzzdiag_i_list[j][j] = Kzz[j][i]
            Kzzdiag_i = ns.stack_mat(Kzzdiag_i_list)

            II_i = - Q * log_gaussian_z(Kzzdiag_i + FF_i_inv, ns) \
                - 0.5 * ns.trace(GG_i.T.dot(FF_i_inv.dot(GPDM.inv_op(FF_i_inv + Kzzdiag_i, ns).dot(FF_i_inv.dot(GG_i))))) \
                + Q * log_gaussian_z(FF_i_inv, ns) \
                + 0.5 * ns.trace(GG_i.T.dot(FF_i_inv.dot(GG_i))) \
                + HH_i
            xt_covars_entropy_i = gaussians_diags_entropy(xt_covars[i], ns=ns)
            elbo += II_i + xt_covars_entropy_i

            pp_FF_list[i] = FF_i
            pp_GG_list[i] = GG_i
            pp_Kzzdiag_list[i] = Kzzdiag_i

        return elbo, (pp_FF_list, pp_GG_list, pp_Kzzdiag_list)


import unittest
import matplotlib.pyplot
class TestRBFKernel(unittest.TestCase):

    # def test_gram(self):
    #     N, D = 5, 3
    #     x1 = np.random.uniform(size=[N, D])
    #     x2 = np.random.uniform(size=[N, D])
    #     sigmasqr = 2.0
    #     width = 3.0

    #     K_uni = RBF_Kern_diag_uniform_x_covar.gram_matrix(sigmasqr, width, x1, x2)
    #     widths = width + np.zeros([D])
    #     K_ARD = RBF_ARD_Kern_diag_x_covar.gram_matrix(sigmasqr, widths, x1, x2)
    #     self.assertTrue(np.allclose(K_ARD, K_uni))
        

    def test_gaussians(self):
        self.assertTrue(gaussian_z(np.array([[2.0]])) \
                == np.sqrt([2.0 * np.pi * 2.0]))
        self.assertTrue(log_gaussian_z(np.array([[2.0]])) \
                == np.log(np.sqrt([2.0 * np.pi * 2.0])))

        N, D = 5, 30
        x1 = np.random.uniform(size=[N, D])
        x2 = np.random.uniform(size=[N, D])
        covars = np.stack([x1.dot(x1.T), x2.dot(x2.T)])
        D = 5
        means = np.random.uniform(size=[2, D])
        xs = np.random.uniform(size=[2, D])
        self.assertTrue(np.allclose([log_gaussian_z(covars[0]), log_gaussian_z(covars[1])],
                log_gaussians_zs(covars)))
        self.assertTrue(np.allclose([log_gaussian_pdf(xs[0], means[0], covars[0]),
                log_gaussian_pdf(xs[1], means[1], covars[1])],
                log_gaussians_pdfs(xs, means, covars)))
        

if __name__ == '__main__':
    unittest.main()

    
