import numpy as np
import theano,theano.ifelse
import theano.tensor as T
import time


# pos def matrix
x=np.random.random((500,500))
x=np.dot(x.T,x)

ts=time.clock()
# Cholesky in pure python for comparison
Lp=np.zeros(x.shape)
for i in range(len(x)):
    for j in range(i+1):
        # cupdate start
        s=x[i, j]
        for k in range(j):
            s -= Lp[i, k] * Lp[j, k]
        if i == j:
            Lp[i, i] = np.sqrt(s)
        else:
            Lp[i, j] = s / Lp[j, j]
        # cupdate end

ppTime=time.clock()-ts

# Cholesky in numpy
ts=time.clock()
Lc=np.linalg.cholesky(x)
npTime=time.clock()-ts

print("Deviation pure python-numpy",np.linalg.norm(Lc-Lp)/np.linalg.norm(Lc))



# Cholesky decomp in Theano !!!

# maximal size of matrix. Theano needs to know this in advance for T.arange to work
MAX_CD_SIZE=10000
# the matrix to decompose
X=T.dmatrix()
# lower cholesky factor matrix
L=T.dmatrix()
# loop indexes
I=T.iscalar()
J=T.iscalar()
# previous elements of the line (index I) that is currently computed
prevElems=T.dvector()


# updating element i,j (cupdate in above python code)
cupdate=lambda J,prevElems,I,L,X:T.set_subtensor(prevElems[J],theano.ifelse.ifelse(T.eq(J,I),T.sqrt(X[I,I]-T.sum(prevElems[:I]*prevElems[:I])),(X[I,J]-T.sum(L[J,:J]*prevElems[:J]))/L[J,J]))
# the loop over i
i_loop,i_updates=theano.scan(
    lambda I,L,X:T.set_subtensor(L[I,:I+1],theano.scan(cupdate,sequences=[T.arange(I+1)],outputs_info=T.zeros(MAX_CD_SIZE),non_sequences=[I,L,X])[0][-1,:I+1]) # the loop over j
                        ,sequences=[T.arange(X.shape[0])],outputs_info=T.zeros_like(X),non_sequences=[X])



CholeskyTheano=theano.function([X],i_loop[-1])

Lt=CholeskyTheano(x)
ts=time.clock()
Lt=CholeskyTheano(x)
thTime=time.clock()-ts

print("Deviation theano-numpy",np.linalg.norm(Lc-Lt)/np.linalg.norm(Lt))
print("Pure python time",ppTime)
print("Numpy python time",npTime)
print("Theano time",thTime)




