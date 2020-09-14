import logging
from six.moves import cPickle
import numpy as np
from scipy import optimize
import numerical.numpytheano.theanopool as tp
import numerical.numpyext.logger as npl


logger = logging.getLogger(__name__)


def decreasing_indexes(f):
    idec = [0]
    fmin = f[0]
    for i, fi in enumerate(f):
        if fi < fmin:
            idec.append(i)
            fmin = fi
    return np.array(idec)


def select_decreasing_negate(f, gf):
    idec = decreasing_indexes(f)
    return -np.array(f)[idec], np.array(gf)[idec]


def save_if_NaN(f, grad, vec, function_and_gradient):
    if np.isnan(f):
        # Save the args state
        filename = "f_nan.pkl"
        print("Critical error: NaN function value. Writing state to {}".format(filename))
        with open(filename, "wb") as filehandle:
            state = function_and_gradient.functions[0].varpool.get_vars_state()
            cPickle.dump(state, filehandle, protocol=cPickle.HIGHEST_PROTOCOL)
        exit()


class OptimizationLog(object):

    def __init__(self, logcallback=None):
        super(OptimizationLog, self).__init__()
        self.logcallback = logcallback
        self.reset()

    def reset(self):
        self.f_df = []  # [(f, df)]
        self.events = []  # [(epoch, event)]

    def log_call(self, f, grad, xvec, function_and_gradient):
        self.f_df.append((float(f), np.sqrt(np.sum(np.array(grad)**2))))
        if self.logcallback is not None:
            self.logcallback(f, grad, xvec, function_and_gradient)

    def log_iteration(self, xvec):
        self.log_event("iteration")

    def log_event(self, event):
        self.events.append((len(self.f_df), event))

    def plot(self, transformation=None):
        import matplotlib.pyplot as plt
        f, df = zip(*self.f_df)
        if transformation is not None:
            f, df = transformation(f, df)
        plt.plot(f, label="f(*)")
        plt.plot(df, label="|grad|")
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.title("Optimization log")
        plt.show()

    def save_to(self, filename):
        with open(filename, "wb") as filehandle:
            cPickle.dump([self.f_df, self.events], filehandle,
                         protocol=cPickle.HIGHEST_PROTOCOL)

    def load_from(self, filename):
        with open(filename, "rb") as filehandle:
            [self.f_df, self.events] = cPickle.load(filehandle)


class KeywordArg(object):

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def to_tuple(self):
        return (self.name, self.value)

    def to_dictionary(self):
        return {self.name: self.value}

    def add_to_dictionary(self, dict):
        dict.update(self.to_dictionary)


class StoppingCriterion(KeywordArg):
    pass


class MaxIterations(StoppingCriterion):

    def __init__(self, maxiter=100):
        super(MaxIterations, self).__init__("maxiter", maxiter)


class PrecisionFactor(StoppingCriterion):
    """
    Set factr=1e4 or lower for better precision.
    Default is 1e7
    """

    def __init__(self, factr=1e7):
        super(PrecisionFactor, self).__init__("factr", factr)


class DisplayProgress(KeywordArg):
    """
    "disp" keyword.
    Default is 0, no output
    """

    def __init__(self, disp=100):
        super(DisplayProgress, self).__init__("disp", disp)



def CallBack(KeywordArg):
    def __init__(self, callback):
        super(CallBack, self).__init__("callback", callback)


def keyword_args_to_dict(kwargs):
    if kwargs is None:
        return {}
    else:
        return {arg.name: arg.value for arg in kwargs}

def optimize_bfgs_l(f_and_df, x0, bounds, varargs=None, disp=0):
    """
    Parameters
    ----------
    f_and_df : callable f_and_df(x)
        returns a tuple (f, df)
    x0 : vector
        start point
    bounds : vector
    varargs : list of additional arguments (like StoppingCriterion(), CallBack())
    Returns
    -------
    xOpt : vector
        optimal point
    f : float
        optimal value
    d : dictionary
        additional info
    """
    niter = [0]  # it is a python thing
    if isinstance(f_and_df, tp.CurriedFunctions):
        if disp > 0:
            logger.info("Optimizing variables: {}".format([arg_var.symbol.name for arg_var in f_and_df.functions[0].args_vars]))
        def callwrapper(xvec):
            res = f_and_df.vectorized(xvec)
            logger.debug(npl.NPRecord("f, |df|", np.array([float(res[0]), np.linalg.norm(res[1:])])))
            if abs(float(res[0])) > 1.0e15:
                logger.debug("Function value is out of range: {}".format(res[0]))
                raise ArithmeticError("Function value is out of range: {}".format(res[0]))
            return res
    elif isinstance(f_and_df, tp.CurriedFunction):
        if disp > 0:
            logger.info("Optimizing variables: {}".format([arg_var.symbol.name for arg_var in f_and_df.args_vars]))
        def callwrapper(xvec):
            res = f_and_df.vectorized(xvec)
            if abs(float(res[0])) > 1.0e15:
                logger.debug("Function value is out of range: {}".format(res[0]))
                raise ArithmeticError("Function value is out of range: {}".format(res[0]))
            return res[0], res[1:]

    def interationcallback(xvec):
        niter[0] += 1
        logger.debug("Iteration {} finished".format(niter[0]))

    kwd = keyword_args_to_dict(varargs)
    xOpt, f, d = optimize.fmin_l_bfgs_b(callwrapper,
                                        x0=x0,
                                        bounds=bounds,
                                        callback=interationcallback,
                                        iprint=1,
                                        **kwd)
    logger.info("Optimized warnflag = {}".format(d['warnflag']))
    logger.info("Optimized task message = {}".format(d['task']))
    logger.info("Optimized f = {}".format(f))
    logger.info("Optimized |df| = {}".format(np.linalg.norm(d['grad'])))
    return xOpt, f, d


def theano_optimize_bfgs_l(f_and_df, varargs=None):
    assert isinstance(f_and_df, (tp.CurriedFunction, tp.CurriedFunctions))
    
    xOpt, f, d = optimize_bfgs_l(f_and_df,
                                 x0=f_and_df.get_args_vals_vector(),
                                 bounds=f_and_df.get_args_bounds_vector(),
                                 varargs=varargs)
    f_and_df.set_args_vals_vector(xOpt)
    return xOpt, f, d



if __name__ == "__main__":
    import sys
    import theano.tensor as tt
    logger.setLevel(logging.DEBUG)
    f = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(f)
    logger.addHandler(h)

    pool = tp.TheanoVarPool()
    k = pool.scalar("k", 2.0, bounds=(1, 10))
    a = pool.matrix("a", -1.0 * np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    b = pool.matrix("b", 1.0 * np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    f_and_df = pool.make_function_and_gradient(tt.sum(k * (a-b)**2), args=[k, a])
    theano_optimize_bfgs_l(f_and_df, varargs=[MaxIterations(1000), PrecisionFactor(10)])
    assert np.allclose(pool.vars[a].get_value(), pool.vars[b].get_value())


