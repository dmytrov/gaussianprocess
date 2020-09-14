import theano
import theano.tensor as T
import theano.tensor.nlinalg as nlinalg
import theano.gof as gof
import numpy as np
import numerical.numpyext.linalg as ntl


class CholeskyInvJitterOp(theano.Op):
    __props__ = ('lower', 'destructive')

    def __init__(self, lower=True, maxiter=10):
        self.lower = lower
        self.maxiter = maxiter
        self.destructive = False

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        assert x.ndim == 2
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        z = outputs[0]
        z[0] = self._cholesky_inv_jitter(x).astype(x.dtype)

    def grad(self, inputs, gradients):
        x, = inputs
        xi = self(x)
        gz, = gradients
        return [-nlinalg.matrix_dot(xi, gz.T, xi).T]

    def _cholesky_inv_jitter(self, x):
        return ntl.cholesky_inv_jitter(x, self.maxiter)

inv_jitter = CholeskyInvJitterOp()


class CholeskyLogDetJitterOp(theano.Op):
    __props__ = ('lower', 'destructive')

    def __init__(self, lower=True, maxiter=10):
        self.lower = lower
        self.maxiter = maxiter
        self.destructive = False

    def infer_shape(self, node, shapes):
        return [()]

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        assert x.ndim == 2
        o = theano.tensor.scalar(dtype=x.dtype)
        return gof.Apply(self, [x], [o])

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        z = outputs[0]
        L = ntl.cholesky_jitter(x, self.maxiter).astype(x.dtype)
        z[0] = np.asarray(2.0 * np.sum(np.log(np.diag(L))), dtype=x.dtype)
        #print("CholeskyLogDetJitterOp.perform", z[0])
        if np.isnan(z[0]):
            print("Error: CholeskyLogDetJitterOp.perform(...) returns NaN")
            print("X: {}".format(x))
            print("X.shape: {}".format(x.shape))
        
    def grad(self, inputs, gradients):
        x, = inputs
        xi = inv_jitter(x)
        gz, = gradients
        return [gz * xi.T]

log_det_jitter = CholeskyLogDetJitterOp()

