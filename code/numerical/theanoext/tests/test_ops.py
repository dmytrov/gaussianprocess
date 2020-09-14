import theano
import theano.tensor as tt
import theano.tensor.nlinalg as nlinalg
import numerical.theanoext.operations.cholesky as cholesky
import numpy as np


X = tt.matrix("X", dtype=theano.config.floatX)
def traceinvcholeskygramm(A):
    return nlinalg.trace(cholesky.inv(tt.dot(A, A.T)))
funcgg = theano.function([X], traceinvcholeskygramm(X))
funcgradgg = theano.function([X], theano.grad(traceinvcholeskygramm(X), X))

x = np.asarray([[1.0, 3.0, 0.0],
                [1.0, 4.0, 3.0],
                [2.1, 5.0, 3.0]], dtype=theano.config.floatX)
tt.verify_grad(traceinvcholeskygramm, (x,), rng=np.random.RandomState(42))
