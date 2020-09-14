import numpy as np
import numerical.numpyext as npx
import numerical.numpyext.linalg as ntla
import matplotlib.pyplot as plt
import ml.gptheano.vecgpdm.equations as eqt
import ml.gptheano.vecgpdm.kernels as krn
import numerical.numpytheano.theanopool as tp
import utils.numerical.integrate as integ
import unittest
import matplotlib.pyplot


class TestRBFKernel(unittest.TestCase):

    def test_quadratic_form(self):
        X = np.array([[0, 1], [1, 2], [5, 5]])
        means = np.array([[0, 0], [1, 1]])
        measure = np.array([[1, 0], [0, 1]])
        self.assertTrue(np.allclose(
                ntla.quadratic_form(X, measure), 
                [1, 5, 50]))
        self.assertTrue(np.allclose(
                ntla.quadratic_form(X, measure, means), 
                [[1, 1], [5, 1], [50, 32]]))


    def test_scan(self):
        A = np.array([[1, 2], [3, 4]])
        k = 10
        result = 1
        for i in range(k):
            result = result * A
        res_np = result

        result = npx.scan(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=np.ones_like(A),
            non_sequences=A,
            n_steps=k)
        res_scan = result[-2][-1]
        
        self.assertTrue(np.all(res_np == res_scan))


    def _test_psi_stats(self, k_func, f_kern, f_psi0, f_psi1, f_psi2, rtol=0.001):
        N, M, D = 4, 3, 2
        x_means = np.reshape(np.random.uniform(size=N*D), [N, D])
        x_covars = np.reshape(np.random.uniform(low = 1, high=2, size=N*D), [N, D])
        aug_z = np.reshape(np.random.uniform(size=M*D), [M, D])
        
        f_gauss = lambda x0, x1, mu, covars: \
                (1.0 / np.sqrt((2.0 * np.pi)**2 * covars[0] * covars[1])) * \
                np.exp(-0.5*((x0 - mu[0])**2 / covars[0] + (x1 - mu[1])**2 / covars[1]))

        f_func_vec = lambda vx0, vx1: k_func(vx0[0], vx0[1], vx1[0], vx1[1])
        K = np.array([[f_func_vec(x0, x1) for x1 in x_means] for x0 in x_means])
        self.assertTrue(np.allclose(K, f_kern(x_means)))
        
        psi0 = f_psi0(x_means, x_covars)
        for n in range(N):     
            func_psi_0 = lambda x0, x1: k_func(x0, x1, x0, x1) \
                    * f_gauss(x0, x1, x_means[n, :], x_covars[n, :])
            s_numerical = integ.integrate(func_psi_0, 
                    ranges=[(-15.0, 15.0), (-15.0, 15.0)], step=0.01)
            self.assertTrue(np.allclose(s_numerical, psi0[n], rtol=rtol))

        psi1 = f_psi1(aug_z, x_means, x_covars)
        for n in range(N):
            for m in range(M):
                func_psi_1 = lambda x0, x1: \
                        k_func(x0, x1, aug_z[m, 0], aug_z[m, 1]) \
                        * f_gauss(x0, x1, x_means[n, :], x_covars[n, :])
                s_numerical = integ.integrate(func_psi_1, 
                        ranges=[(-15.0, 15.0), (-15.0, 15.0)], step=0.01)
                self.assertTrue(np.allclose(s_numerical, psi1[n, m], rtol=rtol))
    
        m_prime = 1
        psi2 = f_psi2(aug_z, x_means, x_covars)
        for n in range(N):
            for m in range(M):
                func_psi_2 = lambda x0, x1: \
                        k_func(x0, x1, aug_z[m, 0], aug_z[m, 1]) \
                        * k_func(x0, x1, aug_z[m_prime, 0], aug_z[m_prime, 1]) \
                        * f_gauss(x0, x1, x_means[n, :], x_covars[n, :])
                s_numerical = integ.integrate(func_psi_2, 
                        ranges=[(-15.0, 15.0), (-15.0, 15.0)], step=0.01)
                self.assertTrue(np.allclose(s_numerical, psi2[n, m, m_prime], rtol=rtol))

    
    def test_psi_stats_linear_ard(self):
        lambdas = np.random.uniform(low = 1, high=2, size=2)
        self._test_psi_stats(
                k_func=lambda x0, x1, y0, y1: x0*y0/lambdas[0] + x1*y1/lambdas[1],
                f_kern=lambda x: eqt.Linear_ARD_Kern_diag_x_covar.gram_matrix(lambdas, x, x),
                f_psi0=lambda m, c: eqt.Linear_ARD_Kern_diag_x_covar.psi_stat_0(lambdas, m, c),
                f_psi1=lambda z, m, c: eqt.Linear_ARD_Kern_diag_x_covar.psi_stat_1(lambdas, z, m),
                f_psi2=lambda z, m, c: eqt.Linear_ARD_Kern_diag_x_covar.psi_stat_2(lambdas, z, m, c))


    def test_psi_stats_rbf_ard(self):
        lambdas = np.random.uniform(low = 1, high=2, size=2)
        sigmasqrf = np.random.uniform(low = 1, high=2, size=1)[0]        
        self._test_psi_stats(
                k_func=lambda x0, x1, y0, y1: sigmasqrf * np.exp(-0.5*((x0 - y0)**2 / lambdas[0] + (x1 - y1)**2 / lambdas[1])),
                f_kern=lambda x: eqt.RBF_ARD_Kern_diag_x_covar.gram_matrix(sigmasqrf, lambdas, x, x),
                f_psi0=lambda m, c: eqt.RBF_ARD_Kern_diag_x_covar.psi_stat_0(sigmasqrf, m),
                f_psi1=lambda z, m, c: eqt.RBF_ARD_Kern_diag_x_covar.psi_stat_1(sigmasqrf, lambdas, z, m, c),
                f_psi2=lambda z, m, c: eqt.RBF_ARD_Kern_diag_x_covar.psi_stat_2(sigmasqrf, lambdas, z, m, c))
                

    def test_psi_stats_rbf_linear_ard(self):
        lambdas_rbf = np.random.uniform(low = 1, high=2, size=2)
        lambdas_linear = np.random.uniform(low = 1, high=2, size=2)
        sigmasqrf = np.random.uniform(low = 1, high=2, size=1)[0]
        self._test_psi_stats(
                k_func=lambda x0, x1, y0, y1:\
                        x0*y0/lambdas_linear[0] + x1*y1/lambdas_linear[1] \
                        + sigmasqrf * np.exp(-0.5*((x0 - y0)**2 / lambdas_rbf[0] + (x1 - y1)**2 / lambdas_rbf[1])),
                f_kern=lambda x: eqt.RBF_plus_Linear_ARD_Kern_diag_x_covar.gram_matrix(sigmasqrf, lambdas_rbf, lambdas_linear, x, x),
                f_psi0=lambda m, c: eqt.RBF_plus_Linear_ARD_Kern_diag_x_covar.psi_stat_0(sigmasqrf, lambdas_linear, m, c),
                f_psi1=lambda z, m, c: eqt.RBF_plus_Linear_ARD_Kern_diag_x_covar.psi_stat_1(sigmasqrf, lambdas_rbf, lambdas_linear, z, m, c),
                f_psi2=lambda z, m, c: eqt.RBF_plus_Linear_ARD_Kern_diag_x_covar.psi_stat_2(sigmasqrf, lambdas_rbf, lambdas_linear, z, m, c))

        

if __name__ == '__main__':
    unittest.main()

    
    
def test_GPFunction():
    ns = tp.NumpyVarPool()
    kobj = krn.RBF_ARD_Kernel(ndims=1, kern_width=0.05, suffix="k1", ns=ns)
    gp = eqt.GP(kobj)
    gpf = gp.sample_function()

    T = 100
    X = np.arange(0.0, 1.0, 1.0/T)[:, np.newaxis] - 0.5
    Y = gpf.sample_values(X)
    X = np.arange(0.0, 1.0, 1.0/T)[:, np.newaxis] * 2 - 1.0
    Y = gpf.sample_values(X)
    X = np.arange(0.0, 1.0, 1.0/T)[:, np.newaxis] * 2 + 1.1
    Y = gpf.sample_values(X)
    plt.figure()
    plt.plot(gpf.sampledX, gpf.sampledY, "x")
    
    kobj = krn.RBF_ARD_Kernel(ndims=2, kern_width=0.55, suffix="k2", ns=ns)
    gp = eqt.GP(kobj, Dx=2, Dy=2)
    gpf = gp.sample_function()
    N = 10  # grid size
    vx, vy = np.meshgrid(np.arange(0.0, 1.0, 1.0/N), 
                         np.arange(0.0, 1.0, 1.0/N))
    X = np.hstack([np.reshape(vx, (N*N, -1)), 
                   np.reshape(vy, (N*N, -1))])
    X = X - 0.45
    Y = gpf.sample_values(X)
    Yr = np.reshape(Y, (N, N, -1))
    plt.figure()
    plt.imshow(Yr[:, :, 0])
    plt.figure()
    plt.imshow(Yr[:, :, 1])
    

def generate_oscillatory_trajectory_2D(T):
    ns = tp.NumpyVarPool()
    
    dt = 0.01 # time discretization
    w1 = 2 * np.pi * 1.0 # base frequency of oscillation in first latent space
    delta1 = 0.3 # dampening

    kobj = krn.RBF_ARD_Kernel(ndims=2, kern_width=1.55, suffix="k2", ns=ns)
    gp = eqt.GP(kobj, Dx=2, Dy=2)
    gpf = gp.sample_function()
    # Supply an oscillatory mean function
    gpf.fmean = lambda X: np.vstack([X[:, 0] + X[:, 1] * dt, \
                                     X[:, 1] - (w1 ** 2 * X[:, 0] + 2.0 * delta1 * w1 * X[:, 1]) * dt]).T

    N = 10  # grid size
    vx, vy = np.meshgrid(np.arange(0.0, 1.0, 1.0/N), 
                         np.arange(0.0, 1.0, 1.0/N))
    X = np.hstack([np.reshape(vx, (N*N, -1)), 
                   np.reshape(vy, (N*N, -1))]) - 0.45
    Y = gpf.sample_values(X)
        
    X = np.array([[1.0, 0.0]])
    for t in range(T):
        xt, _ = gpf.posterior_predictive(X[-1, :])
        X = np.vstack([X, xt])
    #plt.figure()
    #plt.plot(X[:, 0], X[:, 1])
    #plt.figure()
    #plt.plot(X)
    return X
    


def test_loglikelihood_single():
    ns = tp.NumpyVarPool()
    T = 200
    Dx = 2
    np.random.seed(5)
    X1 = generate_oscillatory_trajectory_2D(T)
    plt.figure()
    plt.plot(X1)
    np.random.seed(10)
    
    kobj = krn.RBF_ARD_Kernel(ndims=Dx, kern_width=1.55, suffix="dyn", ns=ns)
    X1in = X1[:-1, :]
    X1out = X1[1:, :]
    Kxx = kobj.gram_matrix(X1in, X1in)
    loglik = eqt.GPDM.loglikelihood_single(Kxx, X1out, 0.1, order=1, ns=ns)
    print(loglik)
    
    Foptimal = eqt.GPDM.F_optimal_single(Kxx, X1out, 0.1, order=1, ns=ns)
    plt.figure()
    plt.plot(X1out, "x")
    plt.plot(Foptimal)
    Foptimal = eqt.GPDM.F_optimal_coupled_kron([[Kxx]], [X1out], np.array([[0.1]]), order=1, ns=ns)
    plt.figure()
    plt.plot(X1out, "x")
    plt.plot(Foptimal[0][0])
    
    
def test_loglikelihood_coupled():
    ns = tp.NumpyVarPool()
    T = 300
    Dx = 2
    np.random.seed(5)
    X1 = generate_oscillatory_trajectory_2D(T)[100:, :]
    plt.figure()
    plt.plot(X1)
    np.random.seed(10)
    X2 = generate_oscillatory_trajectory_2D(T)[100:, :]
    plt.figure()
    plt.plot(X2)
    
    X1in = X1[:-1, :]
    X1out = X1[1:, :]
    X2in = X2[:-1, :]
    X2out = X2[1:, :]
    Xin = [X1in, X2in]
    Xout = [X1out, X2out]
    kobjs = [[krn.RBF_ARD_Kernel(ndims=Dx, kern_width=1.55, suffix="dyn11", ns=ns),
              krn.RBF_ARD_Kernel(ndims=Dx, kern_width=1.55, suffix="dyn12", ns=ns)],
             [krn.RBF_ARD_Kernel(ndims=Dx, kern_width=4.55, suffix="dyn21", ns=ns),
              krn.RBF_ARD_Kernel(ndims=Dx, kern_width=4.55, suffix="dyn22", ns=ns)]]
    Kxxs = [[kobjs[0][0].gram_matrix(X1in, X1in), kobjs[0][1].gram_matrix(X1in, X1in)],
            [kobjs[1][0].gram_matrix(X2in, X2in), kobjs[1][1].gram_matrix(X2in, X2in)]]
    alphas = np.array([[0.1, 1.1], 
                       [1.1, 0.1]])
    loglik = eqt.GPDM.loglikelihood_coupled_marginalized(Kxxs, [X1out, X2out], alphas, order=1, ns=ns)
    print(loglik)
    
    Foptimal = eqt.GPDM.F_optimal_coupled_kron(Kxxs, [X1out, X2out], alphas, order=1, ns=ns)
    plt.figure()
    plt.title("Foptimal[0][0] vs X1out")
    plt.plot(Foptimal[0][0])
    plt.plot(X1out, "x")
    plt.figure()
    plt.title("Foptimal[1][0] vs X1out")
    plt.plot(Foptimal[1][0])
    plt.plot(X1out, "x")
    
    gpdyn = eqt.matrix_of_nones(2, 2)
    for i in range(2):
        for j in range(2):
            gp = eqt.GP(kobjs[j][i], Dx=2, Dy=2)
            gpdyn[j][i] = gp.sample_function()
            gpdyn[j][i].condition_on(Xin[j], Foptimal[j][i])
    
    X = eqt.GPDM.generate_mean_prediction_single(gpdyn[0][0], x0=Xin[0][0, :], T=T, order=1)
    X2 = eqt.GPDM.generate_mean_prediction_coupled([[gpdyn[0][0]]], alpha=np.array([[alphas[0, 0]]]), x0s=[Xin[0][0, :]], T=T, order=1)
    plt.figure()
    plt.title("Generated X1 trajectory from Foptimal vs X1in")
    plt.plot(Xin[0][:T, :])
    plt.plot(X, "x")
    plt.plot(X2[0], "+")
    
    Xs = eqt.GPDM.generate_mean_prediction_coupled(gpdyn, alpha=alphas, x0s=[Xin[0][0, :], Xin[1][0, :]], T=T, order=1)
    plt.figure()
    plt.title("Generated X1 X2 coupled trajectories")
    plt.plot(Xin[0][:T, :])
    plt.plot(Xin[1][:T, :])
    plt.plot(Xs[0], "x")
    plt.plot(Xs[1], "+")
    
    
    
def test_gaussin_PoE():
    means = np.array([[0.0], [2.0]])
    covars = np.array([[2.0, 5.0],
                       [2.0, 2.0]])
    m, c = eqt.gaussian_PoE(means, covars)
    print("mean: {}, covar: {}".format(m, c))
    
