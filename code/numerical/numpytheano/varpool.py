from __future__ import print_function
import time
import itertools
import collections
from six.moves import cPickle
import numpy as np
from scipy import optimize
import theano
import theano.tensor as tt
import theano.compile.sharedvalue as ts
import numerical.numpytheano as nt
import numerical.numpyext.vectorization as ve
import utils.debug.processinfo as dpi



def expr_tree_leafs(expr):  # even though it is "leaves"
    if expr.owner is None:
        return [expr]
    res = []
    for input in expr.owner.inputs:
        res.extend(expr_tree_leafs(input))
    return res


def unique(lst):
    return list(set(lst))


class TheanoTerminalVar(object):
    def __init__(self, symbol, value=None, bounds=None, tags=None):
        super(TheanoTerminalVar, self).__init__()
        assert (value is not None) or (isinstance(symbol, ts.SharedVariable))
        self.symbol = symbol
        self._value = None
        self.bounds = (None, None) if bounds is None else bounds
        if len(self.bounds) != 2:
            for bound in self.bounds:
                assert bound[0] <= bound[1]
        self.tags = set()
        if tags is not None:
            if not isinstance(tags, collections.Iterable):
                tags = (tags,)
            self.tags = set(tags)
        self.set_value(value)

    def get_value(self):
        if isinstance(self.symbol, ts.SharedVariable):
            return self.symbol.get_value()
        else:
            if isinstance(self._value , np.ndarray):
                return self._value.copy()
            else:
                return self._value

    def set_value(self, value):
        # TODO: check bounds violation
        if np.any(np.isnan(value)):
            print("Attempted to assing NaN to '{}', ignoring".format(self.get_name()))
            return
                
        if isinstance(self.symbol, ts.SharedVariable):
            return self.symbol.set_value(value)
        else:
            if isinstance(value, np.ndarray):
                self._value = value.copy()
            else:
                self._value = value

    def get_name(self):
        return self.symbol.name

    def get_args_bounds_vector(self):
        value = self.get_value()
        if isinstance(value, np.ndarray):
            if self.bounds == (None, None):
                return (self.bounds,) * value.size
            elif len(self.bounds) == value.size and isinstance(self.bounds[0], collections.Iterable):
                return self.bounds
            else:
                noise = 0.001 * (self.bounds[1]-self.bounds[0]) * np.random.uniform(size=value.size)
                return [(self.bounds[0] + noise[i], self.bounds[1] + noise[i]) for i in range(value.size)]
        else:
            return (self.bounds,)


class NumpyTerminalVar(np.ndarray):
    def __new__(cls, input_array, name, value=None, bounds=None, tags=None):
        np_array = np.asarray(input_array)
        obj = np_array.view(cls)
        obj.name = name
        obj.bounds = (None, None) if bounds is None else bounds
        obj.symbol = np_array
        if len(obj.bounds) != 2:
            for bound in obj.bounds:
                assert bound[0] <= bound[1]
        obj.tags = set()
        if tags is not None:
            if not isinstance(tags, collections.Iterable):
                tags = (tags,)
            obj.tags = set(tags)
        obj.set_value(value)
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.name = getattr(obj, "name", None)
        self.bounds = getattr(obj, "bounds", None)
        self.tags = getattr(obj, "tags", None)
        self.symbol = getattr(obj, "symbol", None)

    def get_value(self):
        return self

    def set_value(self, value):
        if isinstance(value , np.ndarray) and len(value.shape) > 0:
            self[:] = value.copy()[:]
        else:
            self.fill(value)

    def get_name(self):
        return self.name

    def get_args_bounds_vector(self):
        value = self.get_value()
        if isinstance(value, np.ndarray):
            if self.bounds == (None, None):
                return (self.bounds,) * value.size
            elif len(self.bounds) == value.size:
                return self.bounds
            else:
                noise = 0.01 * (self.bounds[1]-self.bounds[0]) * np.random.uniform(size=value.size)
                return [(self.bounds[0] + noise[i], self.bounds[1] + noise[i]) for i in range(value.size)]
        else:
            return (self.bounds,)


class CurriedFunction(object):
    def __init__(self, function, args_vars, curried_vars, varpool=None):
        super(CurriedFunction, self).__init__()
        self.function = function
        self.args_vars = list(args_vars)
        self.curried_vars = list(curried_vars)
        self.set_args_vals_on_call = True
        self.varpool = varpool
        
    def __call__(self, *args):
        if self.set_args_vals_on_call:
            self.set_args_vals(args)
        curried_vals = [var.get_value() for var in self.curried_vars]
        all_vals = list(args) + curried_vals
        if isinstance(self.function, collections.Iterable):
            return [function(*all_vals) for function in self.function]
        else:
            return self.function(*all_vals)

    def vectorized(self, vec=None):
        if vec is None:
            vec = []
        args = ve.unvectorize(vec, templates=[var.get_value() for var in self.args_vars])
        if self.set_args_vals_on_call:
            self.set_args_vals(args)
        res = ve.vectorize(self(*args))
        return res

    def set_args_vals(self, args_vals):
        for var, val in zip(self.args_vars, args_vals):
            var.set_value(val)

    def get_args_vals_vector(self):
        return ve.vectorize([var.get_value() for var in self.args_vars])

    def set_args_vals_vector(self, vec):
        args = ve.unvectorize(vec, templates=[var.get_value() for var in self.args_vars])
        self.set_args_vals(args)

    def get_args_bounds_vector(self):
        return list(itertools.chain.from_iterable([var.get_args_bounds_vector() for var in self.args_vars]))

       
class CurriedFunctions(object):
    def __init__(self, functions):
        super(CurriedFunctions, self).__init__()
        self.functions = functions
        
    def __call__(self, *args):
        return [f(*args) for f in self.functions]
        
    def vectorized(self, vec):
        return [f.vectorized(vec) for f in self.functions]

    def set_args_vals(self, args_vals):
        self.functions[0].set_args_vals(args_vals)
        
    def get_args_vals_vector(self):
        return self.functions[0].get_args_vals_vector()

    def set_args_vals_vector(self, vec):
        self.functions[0].set_args_vals_vector(vec)
    
    def get_args_bounds_vector(self):
        return self.functions[0].get_args_bounds_vector()
    

class SymbolFactory(object):
    def scalar(self, name, value, bounds=None, differentiable=None, tags=None):
        raise NotImplementedError()

    def vector(self, name, value, bounds=None, differentiable=None, tags=None):
        raise NotImplementedError()

    def int_vector(self, name, value, bounds=None, tags=None):
        raise NotImplementedError()

    def matrix(self, name, value, bounds=None, differentiable=None, tags=None):
        raise NotImplementedError()

    def get_value(self, terminal_symbol):
        raise NotImplementedError()

    def set_value(self, terminal_symbol, value):
        raise NotImplementedError()

    def evaluate(self, expr):
        raise NotImplementedError()


class NumpyVarPool(SymbolFactory, nt.NumpyLinalg):
    def __init__(self):
        super(NumpyVarPool, self).__init__()
        self.vars = {}  # name->var dictionary

    def add_terminal_var(self, var, differentiable=None):
        assert isinstance(var, NumpyTerminalVar)
        assert var.name not in self.vars
        self.vars[var.name] = var

    def scalar(self, name, value, bounds=None, differentiable=None, tags=None):
        var = NumpyTerminalVar(value, name, value, bounds, tags)
        self.add_terminal_var(var, differentiable)
        return var.symbol  # np.ndarray

    def vector(self, name, value, bounds=None, differentiable=None, tags=None):
        var = NumpyTerminalVar(value, name, value, bounds, tags)
        self.add_terminal_var(var, differentiable)
        return var.symbol  # np.ndarray

    def int_vector(self, name, value, bounds=None, tags=None):
        var = NumpyTerminalVar(value, name, value, bounds, tags)
        self.add_terminal_var(var, False)
        return var.symbol  # np.ndarray

    def matrix(self, name, value, bounds=None, differentiable=None, tags=None):
        var = NumpyTerminalVar(value, name, value, bounds, tags)
        self.add_terminal_var(var, differentiable)
        return var.symbol  # np.ndarray

    def to_var(self, symbol):
        for var in self.vars.values():
            if var.symbol is symbol:
                return var
        return None

    def get_value(self, terminal_symbol):
        return terminal_symbol

    def set_value(self, terminal_symbol, value):
        self.to_var(terminal_symbol).set_value(value)

    def evaluate(self, expr):
        return expr

class TheanoVarPool(SymbolFactory, nt.TheanoLinalg):
    def __init__(self):
        super(TheanoVarPool, self).__init__()
        print("Creating Theano variable pool")
        print(" - theano.config.mode: {}".format(theano.config.mode))
        print(" - theano.config.optimizer: {}".format(theano.config.optimizer))
        print(" - theano.config.floatX: {}".format(theano.config.floatX))
        print(" - theano.config.device: {}".format(theano.config.device))
        print(" - theano.config.openmp: {}".format(theano.config.openmp))
        print(" - theano.config.blas.ldflags: {}".format(theano.config.blas.ldflags))
        self.vars = {}  # symbol->var dictionary
        self.non_differentiable = set()  # set of non-differentiable symbols (indexes e.g.)
        self.function_cache = {}  # {(expr, argsymbols)->function} cache
        self.gradient_cache = {}  # {expr->{(wrts)->gradients}} cache

    def get_vars_state(self):
        return {var.symbol.name:var.get_value() for var in self.vars.values()}

    def set_vars_state(self, vars):
        self_var_by_name = {symbol.name:var for symbol, var in iter(self.vars.items())}
        for name, value in iter(vars.items()):
            self_var_by_name[name].set_value(value)

    def add_terminal_var(self, var, differentiable=None):
        if differentiable is None:
            differentiable = True
        assert isinstance(var, TheanoTerminalVar)
        assert var.symbol not in self.vars
        self.vars[var.symbol] = var
        if not differentiable:
            self.non_differentiable.add(var.symbol)

    def add_terminal_vars(self, vars):
        for var in vars:
            self.add_terminal_var(var)

    def scalar(self, name, value, bounds=None, differentiable=None, tags=None):
        var = TheanoTerminalVar(tt.scalar(name), value, bounds, tags)
        self.add_terminal_var(var, differentiable)
        return var.symbol

    def vector(self, name, value, bounds=None, differentiable=None, tags=None):
        var = TheanoTerminalVar(tt.vector(name), value, bounds, tags)
        self.add_terminal_var(var, differentiable)
        return var.symbol

    def int_vector(self, name, value, bounds=None, tags=None):
        var = TheanoTerminalVar(tt.ivector(name), value, bounds, tags)
        self.add_terminal_var(var, False)
        return var.symbol

    def matrix(self, name, value, bounds=None, differentiable=None, tags=None):
        var = TheanoTerminalVar(tt.matrix(name), value, bounds, tags)
        self.add_terminal_var(var, differentiable)
        return var.symbol

    def to_var(self, symbol_or_var):
        return symbol_or_var if isinstance(symbol_or_var, TheanoTerminalVar) else self.vars[symbol_or_var]
        
    def to_vars(self, symbols_or_vars):
        return [self.to_var(symbol_or_var) for symbol_or_var in symbols_or_vars]

    def to_symbol(self, symbol_or_var):
        return symbol_or_var.symbol if isinstance(symbol_or_var, TheanoTerminalVar) else symbol_or_var

    def to_symbols(self, symbols_or_vars):
        return [self.to_symbol(symbol_or_var) for symbol_or_var in symbols_or_vars]

    def all_args_symbols(self, expr):
        leafs = unique(expr_tree_leafs(expr))
        all_args = [leaf for leaf in leafs if isinstance(leaf, tt.TensorVariable) and not isinstance(leaf, tt.TensorConstant)]
        for arg in all_args:
            assert arg in self.vars
        return all_args

    def curried_symbols(self, expr, args_symbols):
        all_args = self.all_args_symbols(expr)
        curried_args = [arg for arg in all_args if arg not in args_symbols]
        return curried_args

    def create_name_to_var_dict(self):
        return {var.symbol.name:var for var in self.vars.values()}

    def make_function(self, expr, args=None):
        if args is None:
            args = []
        if args == all:
            args = self.all_args_symbols(expr)
        if not isinstance(args, collections.Iterable):
            args = [args]
        for arg in args:
            assert isinstance(arg, (TheanoTerminalVar, tt.TensorType, tt.TensorVariable))
        args_symbols = [arg for arg in self.to_symbols(args) if arg not in self.non_differentiable]
        args_vars = self.to_vars(args_symbols)
        curried_symbols = self.curried_symbols(expr, args_symbols)
        curried_vars = self.to_vars(curried_symbols)
        key = (expr,) + tuple(args_symbols)
        if key not in self.function_cache:
            #print("Compiling theano function")
            f = theano.function(args_symbols + curried_symbols, expr, allow_input_downcast=True)
            self.function_cache[key] = f
        return CurriedFunction(function=self.function_cache[key], args_vars=args_vars, curried_vars=curried_vars, varpool=self)

    def make_gradient(self, expr, args=None):
        if args is None:
            args = []
        if args == all:
            args = self.all_args_symbols(expr)
        if not isinstance(args, collections.Iterable):
            args = [args]
        for arg in args:
            assert isinstance(arg, (TheanoTerminalVar, tt.TensorType, tt.TensorVariable))
        args_symbols = [arg for arg in self.to_symbols(args) if arg not in self.non_differentiable]
        args_vars = self.to_vars(args_symbols)
        curried_symbols = self.curried_symbols(expr, args_symbols)
        curried_vars = self.to_vars(curried_symbols)
        all_args_symbols = args_symbols + curried_symbols
        if expr not in self.gradient_cache:
            self.gradient_cache[expr] = {}
        cached_gradients = self.gradient_cache[expr]

        wrts_key = tuple(args_symbols)
        if wrts_key not in cached_gradients:
            #dpi.print_meminfo("Before compiling gradients")
            new_gradient = theano.function(all_args_symbols,
                                           theano.grad(expr, args_symbols),
                                           allow_input_downcast=True)
            self.gradient_cache[expr][wrts_key] = new_gradient
            #dpi.print_meminfo("After compiling gradients")

        return CurriedFunction(function=self.gradient_cache[expr][wrts_key],
                               args_vars=args_vars, curried_vars=curried_vars, varpool=self)


    def make_function_and_gradient(self, expr, args=None):
        return CurriedFunctions([self.make_function(expr, args), self.make_gradient(expr, args)])

    def get_value(self, terminal_symbol):
        return self.vars[self.to_symbol(terminal_symbol)].get_value()

    def set_value(self, terminal_symbol, value):
        self.vars[self.to_symbol(terminal_symbol)].set_value(value)

    def evaluate(self, expr):
        """
        Syntactic sugar
        """
        return self.make_function(expr)()


def select_decreasing_negate(f, gf):
    ires = [0]
    fmin = f[0]
    for i, fi in enumerate(f):
        if fi < fmin:
            ires.append(i)
            fmin = fi
    ires = np.array(ires)
    return -np.array(f)[ires], np.array(gf)[ires]

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
        self.val = []
        self.grad = []

    def reset(self):
        self.val = []
        self.grad = []

    def log(self, f, grad, vec, function_and_gradient):
        self.val.append(float(f))
        self.grad.append(np.sqrt(np.sum(np.array(grad)**2)))
        if self.logcallback is not None:
            self.logcallback(f, grad, vec, function_and_gradient)

    def plot(self, transformation=None):
        import matplotlib.pyplot as plt
        if transformation is None:
            val, grad = self.val, self.grad
        else:
            val, grad = transformation(self.val, self.grad)
        plt.plot(val, label="f(*)")
        plt.plot(grad, label="|grad|")
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.title("Optimization log")
        plt.show()

    def save_to(self, filename):
        with open(filename, "wb") as filehandle:
            cPickle.dump([self.val, self.grad], filehandle, protocol=cPickle.HIGHEST_PROTOCOL)

    def load_from(self, filename):
        with open(filename, "rb") as filehandle:
            [self.val, self.grad] = cPickle.load(filehandle)
    

def optimize_bfgs_l(function_and_gradient, maxiter=100, factr=1e7, logger=None):  # set factr=1e4 or lower for better precision
    assert isinstance(function_and_gradient, CurriedFunctions)
    print("Optimizing variables: {}".format([arg_var.symbol.name for arg_var in function_and_gradient.functions[0].args_vars]))
    #print("x0 = ", function_and_gradient.get_args_vals_vector())
    #print("Bounds = ", function_and_gradient.get_args_bounds_vector())
    def callwrapper(vec):
        res = function_and_gradient.vectorized(vec)
        if logger is not None:
            logger.log(res[0], res[1:], vec, function_and_gradient)
        return res
    xOpt, f, d = optimize.fmin_l_bfgs_b(callwrapper,
                                       x0=function_and_gradient.get_args_vals_vector(),
                                       bounds=function_and_gradient.get_args_bounds_vector(),
                                       disp=1, maxiter=maxiter, factr=factr)
    function_and_gradient.set_args_vals_vector(xOpt)
    

if __name__ == "__main__":
    pool = NumpyVarPool()
    k = pool.scalar("k", 2.0, bounds=(1, 10))
    a = pool.matrix("a", -1.0 * np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    b = pool.matrix("b", 1.0 * np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    c = k**2 * (a - b)**2
    print(pool.evaluate(c))

    pool = TheanoVarPool()
    k = pool.scalar("k", 2.0, bounds=(1, 10))
    a = pool.matrix("a", -1.0 * np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    b = pool.matrix("b", 1.0 * np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    c = k**2 * (a - b)**2
    print(pool.evaluate(c))
 
    c_func = pool.make_function(c)
    print(c_func())
    print(c_func.vectorized())
    
    c_func_b = pool.make_function(c, args=[b])
    print(c_func_b(4.0 * np.identity(3)))

    c_func_b = pool.make_function(c, args=[b])
    print(c_func_b(4.0 * np.identity(3)))

    vec = 2 * c_func_b.get_args_vals_vector()
    print(c_func_b.vectorized(vec))
    
    print(c_func_b.get_args_vals_vector())
    c_func_b.set_args_vals_on_call = True
    print(c_func_b.vectorized(vec))
    print(c_func_b.get_args_vals_vector())

    f = tt.sum(c)
    print(pool.make_function(f)())
    df_da = pool.make_gradient(f, args=[a])
    print(df_da(4.0 * np.identity(3)))
    df_dab = pool.make_gradient(f, args=[a, b])
    print(df_dab(2.0 * np.identity(2), 3.0 * np.identity(2)))
    print(c_func_b.vectorized(vec))

    f_df_da = pool.make_function_and_gradient(f, args=[a])
    print(f_df_da(4.0 * np.identity(3)))
    print(f_df_da.vectorized(vec))
    print(f_df_da.get_args_bounds_vector())

    logger = OptimizationLog()
    f_df = pool.make_function_and_gradient(tt.sum(k * (a-b)**2), args=[k, a])
    print(pool.vars[k].get_value())
    print(pool.vars[a].get_value())
    print(pool.vars[b].get_value())
    optimize_bfgs_l(f_df, maxiter=100, logger=logger)
    print(pool.vars[k].get_value())
    print(pool.vars[a].get_value())
    print(pool.vars[b].get_value())
    print("Values: ", logger.val)
    print("|grad|: ", logger.grad)
    logger.plot()

    v = pool.vector("v", np.array([1, 0, 0]))
    print(pool.make_function(tt.dot(pool.vars[a].get_value(), a.dot(v)))())
