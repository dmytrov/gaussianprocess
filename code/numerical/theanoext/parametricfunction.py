import itertools
import numpy as np
import theano.tensor as tt
import collections
import numerical.numpyext.vectorization as ve

def symbols(vars):
    return [var.symbolic for var in vars]

def values(vars):
    return [var.val for var in vars]

def get_values_vec(vars):
    return ve.vectorize([var.val for var in vars])

def set_values_vec(vars, vec):
    for var in vars:
        var.val, vec = ve.unvectorizesingle(vec, var.val)

def get_bounds(vars):
    return list(itertools.chain.from_iterable([var.get_bounds() for var in vars]))

def args_list(expr):
    if expr.owner is not None:
        return list(itertools.chain.from_iterable([args_list(input) for input in expr.owner.inputs]))
    if isinstance(expr, tt.TensorVariable):
        return [expr]
    return []

def unique(lst=[]):
    return list(set(lst))

def flatten(tree=[]):
    if not isinstance(tree, collections.Iterable):
        return [tree]
    else:
        return itertools.chain.from_iterable([flatten(x) for x in tree])

    # if not tree:
    #     return []
    # elif not isinstance(tree, collections.Iterable):
    #     return [tree]
    # else:
    #     return itertools.chain.from_iterable([flatten(x) for x in tree])

class SymFuncPair(object):
    """
    A pair of a symbolic expression for Theano and a callable function for NumPy.
     Base abstract class
    """
    def __init__(self):
        self.symbolic = None
        self.function = lambda: None

class Variable(SymFuncPair):
    """
    Base class for all variables
    """
    def __init__(self, sym, val):
        super(Variable, self).__init__()
        self.val = val
        self.symbolic = sym
        self.function = lambda x: x

class FloatVariable(Variable):
    """
    Base class for float variables. Have bounds for optimization constraints
    """
    def __init__(self, sym, val, bounds=(float("-inf"), float("inf"))):
        super(FloatVariable, self).__init__(sym, val)
        self.bounds = bounds

    def get_bounds(self):
        raise NotImplementedError()


class ScalarVariable(FloatVariable):
    """
    Scalar float variable
    """
    def __init__(self, name, val, bounds=(float("-inf"), float("inf"))):
        super(ScalarVariable, self).__init__(tt.scalar(name), val, bounds)

    def get_bounds(self):
        return (self.bounds,)

class VectorVariable(FloatVariable):
    """
    Vector float variable
    """
    def __init__(self, name, val, bounds=(None, None)):
        super(VectorVariable, self).__init__(tt.vector(name), val, bounds)

    def get_bounds(self):
        if self.bounds == (None, None):
            return (self.bounds,) * self.val.size
        else:
            noise = 0.01 * (self.bounds[1]-self.bounds[0]) * np.random.uniform(size=self.val.size)
            return [(self.bounds[0] + noise[i], self.bounds[1] + noise[i]) for i in range(self.val.size)]

class MatrixVariable(FloatVariable):
    """
    Matrix float variable
    """
    def __init__(self, name, val, bounds=(None, None)):
        super(MatrixVariable, self).__init__(tt.matrix(name), val, bounds)

    def get_bounds(self):
        if self.bounds == (None, None):
            return (self.bounds,) * self.val.size
        else:
            noise = 0.01 * (self.bounds[1]-self.bounds[0]) * np.random.uniform(size=self.val.size)
            return [(self.bounds[0] + noise[i], self.bounds[1] + noise[i]) for i in range(self.val.size)]

class IntVectorVariable(Variable):
    """
    Vector of integers (indexes e.g.)
    """
    def __init__(self, name, val=[]):
        super(IntVectorVariable, self).__init__(tt.ivector(name), val)

class ParametricFunction(SymFuncPair):
    def __init__(self):
        super(ParametricFunction, self).__init__()
        self._params = []
        self._args = []
        self._consts = []
        self._children = []

    def get_params(self):
        """
        Carried parameters (parameters of the function)
        :return:
        """
        return unique(flatten(self._params + [child.get_params() for child in self._children]))

    def get_args(self):
        """
        Arguments of the function
        :return:
        """
        return unique(flatten(self._args + [child.get_args() for child in self._children]))

    def get_consts(self):
        """
        Constants used by the function
        :return:
        """
        return unique(flatten(self._consts + [child.get_consts() for child in self._children]))

    def get_all_vars(self):
        """
        All variables the function depends on
        :return:
        """
        return unique(self.get_args() + self.get_params() + self.get_consts())

    def get_all_non_const_vars(self):
        """
        All variables the function depends on
        :return:
        """
        return unique(self.get_args() + self.get_params())

    def value(self, *args):
        return self.function(*args)
