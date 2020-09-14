import numpy as np
import theano
import theano.tensor as tt
import theano.tensor.slinalg as ttslinalg
import numerical.numpyext as npx
import numerical.numpyext.linalg as ntl
import numerical.theanoext as ttx


class LinalgInterface(object):
    pi = np.pi
    newaxis = None

    @staticmethod
    def dot(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def trace(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def choleskyinv(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def det(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def exp(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def log(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def identity(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def zeros(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def diag(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def cholesky(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def ones_like(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def sqrt(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def inv(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def sum(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def prod(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def kron(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def mean(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def concatenate(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def zeros_like(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def scan(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def transpose(*args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def dimshuffle(cls, *args, **kwargs):
        return cls.transpose(*args, **kwargs)

    @staticmethod
    def stack(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def stack_mat(matlist):
        raise NotImplementedError()

    @staticmethod
    def stacklists(matlist):
        raise NotImplementedError()

    @staticmethod
    def log_det(matrix):
        raise NotImplementedError()

    @staticmethod
    def logger(var, callback):
        raise NotImplementedError()

    @staticmethod
    def sin(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def cos(*args, **kwargs):
        raise NotImplementedError()

class NumpyLinalg(LinalgInterface):
    newaxis = np.newaxis

    @staticmethod
    def dot(*args, **kwargs):
        return np.dot(*args, **kwargs)

    @staticmethod
    def trace(*args, **kwargs):
        return np.trace(*args, **kwargs)

    @staticmethod
    def choleskyinv(*args, **kwargs):
        return ntl.cholesky_inv_jitter(*args, **kwargs)

    @staticmethod
    def det(*args, **kwargs):
        return np.linalg.det(*args, **kwargs)

    @staticmethod
    def exp(*args, **kwargs):
        return np.exp(*args, **kwargs)

    @staticmethod
    def log(*args, **kwargs):
        return np.log(*args, **kwargs)

    @staticmethod
    def identity(*args, **kwargs):
        return np.identity(*args, **kwargs)

    @staticmethod
    def zeros(*args, **kwargs):
        return np.zeros(*args, **kwargs)

    @staticmethod
    def diag(*args, **kwargs):
        return np.diag(*args, **kwargs)

    @staticmethod
    def cholesky(*args, **kwargs):
        return np.linalg.cholesky(*args, **kwargs)

    @staticmethod
    def ones_like(*args, **kwargs):
        return np.ones_like(*args, **kwargs)

    @staticmethod
    def sqrt(*args, **kwargs):
        return np.sqrt(*args, **kwargs)

    @staticmethod
    def inv(*args, **kwargs):
        return np.linalg.inv(*args, **kwargs)

    @staticmethod
    def sum(*args, **kwargs):
        return np.sum(*args, **kwargs)

    @staticmethod
    def prod(*args, **kwargs):
        return np.prod(*args, **kwargs)

    @staticmethod
    def kron(*args, **kwargs):
        return np.kron(*args, **kwargs)

    @staticmethod
    def mean(*args, **kwargs):
        return np.mean(*args, **kwargs)

    @staticmethod
    def concatenate(*args, **kwargs):
        return np.concatenate(*args, **kwargs)

    @staticmethod
    def zeros_like(*args, **kwargs):
        return np.zeros_like(*args, **kwargs)

    @staticmethod
    def scan(*args, **kwargs):
        return npx.scan(*args, **kwargs)

    @staticmethod
    def transpose(*args, **kwargs):
        return args[0].transpose(*(args[1:]), **kwargs)

    @staticmethod
    def stack(*args, **kwargs):
        return np.stack(*args, **kwargs)

    @staticmethod
    def stack_mat(matlist):
        return np.concatenate([np.concatenate(row, axis=1) for row in matlist], axis=0)
        #return np.vstack([np.hstack(row) for row in matlist])

    @staticmethod
    def stacklists(matlist):
        return np.concatenate([item[np.newaxis] for item in (np.concatenate([item[np.newaxis] for item in row], axis=0) for row in matlist)], axis=0)

    @staticmethod
    def log_det(matrix):
        if len(matrix.shape) == 2:
            return 2.0 * np.sum(np.log( np.diag(np.linalg.cholesky(matrix))))
        elif len(matrix.shape) == 3:
            return [2.0 * np.sum(np.log( np.diag(np.linalg.cholesky(m)))) for m in matrix]

    @staticmethod
    def logger(var, callback=ttx.operations.logger.log_to_screen):
        callback(var)
        return var

    @staticmethod
    def sin(*args, **kwargs):
        return np.sin(*args, **kwargs)

    @staticmethod
    def cos(*args, **kwargs):
        return np.cos(*args, **kwargs)


class TheanoLinalg(LinalgInterface):
    newaxis = np.newaxis

    @staticmethod
    def dot(*args, **kwargs):
        return tt.dot(*args, **kwargs)

    @staticmethod
    def trace(*args, **kwargs):
        assert len(args) == 1 or len(args) == 3
        if len(args) == 3:
            assert (args[1] == 1 or args[1] == 2) and (args[2] == 1 or args[2] == 2) and args[2] + args[2] == 3
        x = args[0]
        if x.ndim == 2:
            return tt.nlinalg.trace(*args, **kwargs)
        elif x.ndim ==3:
            xtrace, _ = theano.scan(fn=lambda m, prev: tt.nlinalg.trace(m),
                                  outputs_info=tt.nlinalg.trace(x[0,:,:]),
                                  sequences=x)
            return xtrace
        else:
            raise NotImplementedError()

    @staticmethod
    def choleskyinv(*args, **kwargs):
        return ttx.cholesky.inv_jitter(*args, **kwargs)

    @staticmethod
    def det(*args, **kwargs):
        assert len(args) == 1
        x = args[0]
        if x.ndim == 0:
            return x
        elif x.ndim == 2:
            return tt.nlinalg.det(*args, **kwargs)
        elif x.ndim ==3:
            xdet, _ = theano.scan(fn=lambda m, prev: tt.nlinalg.det(m),
                                  outputs_info=tt.nlinalg.det(x[0,:,:]),
                                  sequences=x)
            return xdet
        else:
            raise NotImplementedError()

    @staticmethod
    def exp(*args, **kwargs):
        return tt.exp(*args, **kwargs)

    @staticmethod
    def log(*args, **kwargs):
        return tt.log(*args, **kwargs)

    @staticmethod
    def identity(*args, **kwargs):
        return tt.eye(*args, **kwargs)

    @staticmethod
    def zeros(*args, **kwargs):
        return tt.zeros(*args, **kwargs)

    @staticmethod
    def diag(*args, **kwargs):
        return tt.diag(*args, **kwargs)

    @staticmethod
    def cholesky(*args, **kwargs):
        return ttslinalg.cholesky(*args, **kwargs)

    @staticmethod
    def ones_like(*args, **kwargs):
        return tt.ones_like(*args, **kwargs)

    @staticmethod
    def sqrt(*args, **kwargs):
        return tt.sqrt(*args, **kwargs)

    @staticmethod
    def inv(*args, **kwargs):
        assert len(args) == 1
        x = args[0]
        if x.ndim == 2:
            return tt.nlinalg.matrix_inverse(*args, **kwargs)
        elif x.ndim ==3:
            xinv, _ = theano.scan(fn=lambda m, prev: tt.nlinalg.matrix_inverse(m),
                                  outputs_info=tt.zeros_like(x[0,:,:]),
                                  sequences=x)
            return xinv
        else:
            raise NotImplementedError()

    @staticmethod
    def sum(*args, **kwargs):
        return tt.sum(*args, **kwargs)

    @staticmethod
    def prod(*args, **kwargs):
        return tt.prod(*args, **kwargs)

    @staticmethod
    def kron(*args, **kwargs):
        assert len(args) == 2
        a = args[0]
        b = args[1]
        assert b.ndim == 2
        if a.ndim == 2:
            return tt.slinalg.kron(a, b)
        elif a.ndim ==3:
            a_kron_b, _ = theano.scan(fn=lambda m, prev, b: tt.slinalg.kron(m, b),
                                  sequences=a,
                                  outputs_info=tt.slinalg.kron(a[0,:,:], b),
                                  non_sequences=b)
            return a_kron_b
        else:
            raise NotImplementedError()

    @staticmethod
    def mean(*args, **kwargs):
        return tt.mean(*args, **kwargs)

    @staticmethod
    def concatenate(*args, **kwargs):
        return tt.concatenate(*args, **kwargs)

    @staticmethod
    def zeros_like(*args, **kwargs):
        return tt.zeros_like(*args, **kwargs)

    @staticmethod
    def scan(*args, **kwargs):
        return theano.scan(*args, **kwargs)

    @staticmethod
    def transpose(*args, **kwargs):
        return args[0].dimshuffle(*(args[1:]), **kwargs)

    @staticmethod
    def stack(*args, **kwargs):
        return tt.stack(*args, **kwargs)

    @staticmethod
    def stack_mat(matlist):
        return tt.concatenate([tt.concatenate(row, axis=1) for row in matlist], axis=0)

    @staticmethod
    def stacklists(matlist):
        return tt.stacklists(matlist)

    @staticmethod
    def log_det(matrix):
        return ttx.cholesky.log_det_jitter(matrix)

    @staticmethod
    def logger(var, callback=ttx.operations.logger.log_to_screen):
        return ttx.logger.log_to_screen(var)

    @staticmethod
    def sin(*args, **kwargs):
        return tt.sin(*args, **kwargs)

    @staticmethod
    def cos(*args, **kwargs):
        return tt.cos(*args, **kwargs)
