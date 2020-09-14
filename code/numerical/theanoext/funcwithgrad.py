import theano
import numerical.numpyext.vectorization as ve

def without(lst, x):
    res = lst
    res.remove(x)
    return res

class FuncWithGrad(object):
    def __init__(self, expr, args, wrts=[]):
        self.expr = expr
        self.args = args
        self.argvals = [None] * len(args)
        self.function = theano.function(args, expr,allow_input_downcast=True)  #, on_unused_input='ignore')
        self.wrts = wrts
        self.cached_gradients = {}  # stored {wrt: gradient} cache
        self.gradients = []  # for wrts
        self.set_wrts(wrts)

    def get_func_value(self, argvals=None):
        if argvals is None:
            argvals = self.get_args_values()

        #print("========================================================================")
        #for i in range(len(self.args)):
        #    if "x_means" in self.args[i].name or "aug_in" in self.args[i].name:
        #        print(self.args[i].name)
        #        print(self.argvals[i][:2, :])
        res = self.function(*argvals)
        #print(res)
        return res

    def get_func_value_vec(self, argvalsvec):
        self.set_args_values_vec(argvalsvec)
        return self.get_func_value()

    def get_func_value_wrt_vec(self, vec):
        # print(vec)
        self.set_wrts_values_vec(vec)
        return self.get_func_value()

    def set_arg_value(self, arg, value):
        self.argvals[self.args.index(arg)] = value

    def set_args_values(self, values):
        self.argvals = values

    def set_args_values_vec(self, vec):
        self.set_args_values(ve.unvectorize(vec, self.argvals))

    def set_wrts(self, wrts):
        self.wrts = list(wrts)
        #w = list(wrts)
        new_wrts = []
        for w in wrts:
            if w not in self.cached_gradients:
                new_wrts += [w]
        new_gradients = [theano.function(self.args, grad,allow_input_downcast=True) for grad in theano.grad(self.expr, new_wrts)]
        self.cached_gradients.update(dict(zip(new_wrts, new_gradients)))
        self.gradients = [self.cached_gradients[wrt] for wrt in wrts]
        #self.gradients = [theano.function(self.args, grad) for grad in theano.grad(self.expr, wrts)]

    def set_wrts_values(self, wrts):
        for i, wrt in enumerate(self.wrts):
            self.argvals[self.args.index(wrt)] = wrts[i]

    def set_wrts_values_vec(self, vec):
        for wrt in self.wrts:
            i = self.args.index(wrt)
            self.argvals[i], vec = ve.unvectorizesingle(vec, self.argvals[i])

    def get_wrt_values_vec(self):
        return ve.vectorize([self.argvals[self.args.index(wrt)] for wrt in self.wrts])

    def get_args_values(self):
        return self.argvals

    def get_arg_value(self, arg):
        return self.argvals[self.args.index(arg)]

    def get_args_values_vec(self):
        return ve.vectorize(self.get_args_values())

    def grad(self, argvals=None):
        if argvals is None:
            argvals = self.get_args_values()
        return [gradient(*argvals) for gradient in self.gradients]

    def grad_vec(self, wrtvals=None):
        if wrtvals is not None:
            self.set_wrts_values_vec(wrtvals)
        return ve.vectorize(self.grad())

    def get_func_value_and_grad_vec(self, wrtvalsvec):
        if wrtvalsvec is not None:
            self.set_wrts_values_vec(wrtvalsvec)
        return self.get_func_value(), self.grad_vec()
