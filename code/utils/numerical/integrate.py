import numpy as np


def integrate(f, ranges, step=0.01):
    """Simple grid integration.

    Args:
        f (callable): The function to integrate with D scalar arguments. 
            Must handle vectorized elementwise computations.

                f(x_1, x-2, ... x_D): R^D -> R
            
        ranges (list): Support for every of D dimansions. 
            A list of suples (min_val, max_val).
            Example for D == 3: 

                ((-1, 1), (-2, 4), (5, 6),)

        step (float): Integration step.

    Returns:
        float: Numerical integration result.

    """
    D = len(ranges)
    ticks = []
    for range in ranges:
        start, stop = range
        ticks.append(np.linspace(start, stop, (stop-start)/step))
    grid = np.meshgrid(*ticks)
    return step**D * np.sum(f(*grid))


def tensors_to_vec(tensors):
    """ Convert list of tensors to a vector.
    
    Args:
        tensors (list): List of tensors.

    Returns:
        vector: All tensors raveled.
    """
    return np.concatenate([np.ravel(t) for t in tensors])


def vec_to_tensors(vec, shapes):
    """ Convert a vector to a list of tensors.
    
    Args:
        vec (vector): Data vector.

        shapes (list): List of tensors shapes.

    Returns:
        list: Ttensors of corresponding shapes.
    """
    sizes = [np.prod(shape) for shape in shapes]
    offsets = np.cumsum([0] + sizes)
    starts = offsets[:-1]
    ends = offsets[1:]
    res = [np.reshape(vec[starts[i]:ends[i]], shape) 
            for i, shape in enumerate(shapes)]
    return res
    

def all_combinations(ticks):
    """ Construct all possible combinations of ticks.
        Example: when ticks are:
            [[0, 1], [1, 2, 3]]

        the created combinations read:
            [[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]]
    
    Args:
        ticks (list): ticks on the axes. 

    Returns:
        list: all possible combinations. 
    """
    if len(ticks) == 0:
        return [[]]   
    else:
        return [[tick] + c for c in all_combinations(ticks[1:]) 
                for tick in ticks[0]]
        

def integrate_vec(f, ranges):
    """Simple grid integration of a vector valued function.

    Args:
        f (callable): The function to integrate with N vector arguments. 
            
                f(X1, X2, ... XN): R^(D1+...+DN) -> R^M
            
        ranges (list): Support for every of N inputs. 
            A list of suples (min_val, max_val, step).
            Example for N == 3, f:R^(1+1+2)->R^M: 

                (
                    ([-1], [1], [0.01]), 
                    ([-2], [4], [0.01]), 
                    ([5, 6], [9, 10], [0.01, 0.01]),
                )

        step (float): Integration step.

    Returns:
        float: Numerical integration result.

    """
    N = len(ranges)
    shapes = [range[0].shape for range in ranges]
    starts = tensors_to_vec([range[0] for range in ranges])
    stops = tensors_to_vec([range[1] for range in ranges])
    steps = tensors_to_vec([range[2] for range in ranges])
    N = len(starts)
    ticks = [np.linspace(start, stop, (stop-start)/step) 
            for start, stop, step in zip(starts, stops, steps)]
    s = 0
    for args in all_combinations(ticks):
        s += f(*vec_to_tensors(args, shapes))
    return s * (np.prod(steps))
    

def gaussian_z(covar):
    """Single-dimensional Gaussian PDF normalizing coefficient.

    Args:
        covar (scalar): Covariance of the Gaussian PDF

    Returns:
        float: Normalizing coefficient Z 
    """
    return np.sqrt(2.0 * np.pi * covar)


def gaussian_pdf(x, mean, covar):
    """Single-dimensional Gaussian PDF.

    Args:
        x (scalar): PDF query point.

        mean (scalar): Mean of the Gaussian PDF.

        covar (scalar): Covariance of the Gaussian PDF.

    Returns:
        float: Probability density.
    """
    return 1.0 / gaussian_z(covar) * np.exp(-0.5*((x-mean)**2)/covar)


def expectation(f_pdf, ranges, step=0.01):
    """ Expectation of a scalar function.
    """
    return integrate(lambda x: x * f_pdf(x), ranges, step)


def gaussian_multivar_z(covar):
    """Multivariate Gaussian PDF normalizing coefficient.

    Args:
        covar (matrix D*D): Covariance of the Gaussian PDF.

    Returns:
        float: Normatizing coefficient.
    """
    return np.sqrt(np.linalg.det(2.0 * np.pi * covar))


def gaussian_multivar_pdf(x, mean, precision, z):
    """Multivariate Gaussian PDF.

    Args:
        x (vector [D]): PDF query point.

        mean (vector[D]): Mean of the Gaussian PDF.

        precision (matrix D*D): precision of the Gaussian PDF.

        z (scalar): Precalculated normalizing coefficient.

    Returns:
        float: Probability density. 
    """
    return 1.0 / z * np.exp(-0.5 * (np.sum((x-mean) * np.dot(precision, (x-mean)))))
    # Slower:
    # return 1.0 / z * np.exp(-0.5 * (np.sum(precision * np.outer((x-mean), (x-mean)))))


def expectation_vec(f_pdf, ranges):
    """ Expectation of a vector function.
    """
    return integrate_vec(lambda x: x * f_pdf(x), ranges)


###############################################################################
###############################################################################
###############################################################################
import unittest

class TestIntegration(unittest.TestCase):

    def test_vertorize(self):
        a = np.array([1, 2, 3, 4])
        b = np.array([[5, 6], [7, 8]])
        c = np.array([9])
        v = tensors_to_vec([a, b, c])
        a1, b1, c1 = vec_to_tensors(v, [a.shape, b.shape, c.shape])
        self.assertTrue(np.allclose(a, a1))
        self.assertTrue(np.allclose(b, b1))
        self.assertTrue(np.allclose(c, c1))


    def test_integrate(self):    
        func = lambda x: gaussian_pdf(x, mean=1.0, covar=1.0)
        s = integrate(func, ranges=[(-10.0, 10.0)], step=0.001)
        self.assertTrue(abs(s - 1.0) < 0.01)


    def test_integrate_multivar(self):    
        mean=np.array([1.0, 2.0])
        covar = np.identity(2)
        precision = np.linalg.inv(covar)
        z = gaussian_multivar_z(covar) 
        func = lambda x: gaussian_multivar_pdf(x, mean, precision, z)
        s = integrate_vec(func, 
                ranges=((
                        np.array([-10.0, -10.0]), 
                        np.array([10.0, 10.0]), 
                        np.array([0.1, 0.1],
                        ),),))
        self.assertTrue(abs(s - 1.0) < 0.01)
        

    def test_expectation(self):        
        mean=np.array([1.0, 2.0])
        covar = np.identity(2)
        precision = np.linalg.inv(covar)
        z = gaussian_multivar_z(covar) 
        func = lambda x: gaussian_multivar_pdf(x, mean, precision, z)
        s = expectation_vec(func, 
                ranges=((
                        np.array([-10.0, -10.0]), 
                        np.array([10.0, 10.0]), 
                        np.array([0.1, 0.1],
                        ),),))
        self.assertTrue(np.allclose(s, mean, rtol=0.01))


if __name__ == '__main__':
    unittest.main()
    