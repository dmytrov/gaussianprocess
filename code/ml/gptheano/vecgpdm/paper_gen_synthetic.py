### Test data generator for vCGPDM learning
from numpy import *
import matplotlib.pyplot as plt
import itertools
import pickle

class GPFunc:
    """Gaussian process function generator"""

    def __init__(self, inducingInputs, meanFunction, kernelFunction):
        """Draw the function
        inducingInputs[M,D] with M: number of inducing points, D: input dimension
        meanFunction(X), kernelFunction(X,X') for X \in R^D"""

        self.X = ndarray((len(inducingInputs),),dtype="object") # transform into object array so we can use numpy's 'outer' ufunc member
        for i in range(len(inducingInputs)): 
            self.X[i] = inducingInputs[i]
    
        self.kernelFunction = lambda X1, X2: frompyfunc(kernelFunction, 2, 1).outer(X1, X2).astype("float64")
        self.meanFunction = lambda X: frompyfunc(meanFunction, 1, 1)(X).astype("float64")
        # draw inducing function values
        self.Kmm = self.kernelFunction(self.X, self.X)
        self.mum = self.meanFunction(self.X)
        self.F = random.multivariate_normal(self.mum, self.Kmm)
        self.Kmmi = linalg.inv(self.Kmm)
        
        self.Kmmi = linalg.inv(self.kernelFunction(self.X, self.X))

    def __call__(self, Xtest):
        """Given Xtest[N,D], return function values"""
        Xtest = atleast_2d(Xtest)
        Xt = ndarray((len(Xtest),), dtype="object")
        for i in range(len(Xt)): 
            Xt[i] = Xtest[i]
        Knm = self.kernelFunction(Xt, self.X)
        return einsum("nm,mp,p->n", Knm, self.Kmmi, (self.F - self.mum)) + self.meanFunction(Xt)

def GP_draw(kern_func, x, size=1):
    return random.multivariate_normal(zeros(x.shape[0]), kern_func(x, x), size=size).T

    
random.seed(250912)
#random.seed(250916)
#random.seed(250920)

TIMESTEPS = 2500
dt = 0.01 # time discretization
# the model has 2 latent spaces, 2D each
w1 = 2 * pi * 1.0 # base frequency of oscillation in first latent space
delta1 = 0.3 # dampening
inducingPoints = vstack([random.uniform(-5.0, 5.0, size=100), random.uniform(-5 * w1, 5 * w1, size=100)]).T
latentMapping11 = GPFunc(inducingPoints,
                         meanFunction=lambda X: X[0] + X[1] * dt,
                         kernelFunction=lambda X1, X2: 0.05 * exp(-1 * linalg.norm(X1 - X2) ** 2))
latentMapping12 = GPFunc(inducingPoints,
                         meanFunction=lambda X: X[1] - (w1 ** 2 * X[0] + 2.0 * delta1 * w1 * X[1]) * dt,
                         kernelFunction=lambda X1, X2: 0.1 * exp(-3.0 / w1 ** 2 * linalg.norm(X1 - X2) ** 2))

w2 = 2 * pi * 0.7 # base frequency of oscillation in second latent space
delta2 = 0.4 # dampening
inducingPoints = vstack([random.uniform(-5.0, 5.0, size=100), random.uniform(-5 * w2, 5 * w2, size=100)]).T
latentMapping21 = GPFunc(inducingPoints,
                         meanFunction=lambda X: X[0] + X[1] * dt,
                         kernelFunction=lambda X1, X2: 0.01 * exp(-0.5 * linalg.norm(X1 - X2) ** 2))
latentMapping22 = GPFunc(inducingPoints,
                         meanFunction=lambda X: X[1] - (w2 ** 2 * X[0] + 2.0 * delta2 * w2 * X[1]) * dt,
                         kernelFunction=lambda X1, X2: 0.01 * exp(-4.0 / w2 ** 2 * linalg.norm(X1 - X2) ** 2))

# create time series in latent spaces
X1 = [array((1.0, 0.0))]
X2 = [array((1.0, 0.0))]
#X2 = [array((0.0,-w2))]
for t in arange(1, TIMESTEPS):
    nextX1 = hstack([latentMapping11(X1[-1]), latentMapping12(X1[-1])])
    #nextX2 = nextX1 * 1.0
    nextX2 = hstack([latentMapping21(X2[-1]), latentMapping22(X2[-1])]) * 0.9 + nextX1 * 0.1
    X1 += [nextX1]
    X2 += [nextX2]

X1 = array(X1)
X2 = array(X2)


# latent-to-observed mappings
inducingPoints = vstack([random.uniform(-5.0, 5.0, size=100), random.uniform(-5 * w1, 5 * w1, size=100)]).T
latentToObserved1 = GPFunc(inducingPoints,
                           meanFunction=lambda X: X[0] * 2.0 + 0.3,
                           kernelFunction=lambda X1, X2:0.01 * exp(-0.5 * linalg.norm(X1 - X2) ** 2))
latentToObserved2 = GPFunc(inducingPoints,
                           meanFunction=lambda X: X[0] ** 3,
                           kernelFunction=lambda X1, X2: 0.01 * exp(-0.5 * linalg.norm(X1 - X2) ** 2))

Y1 = latentToObserved1(Xtest=X1)
Y2 = latentToObserved2(Xtest=X2)

Y1 = GP_draw(kern_func=lambda X1, X2: 0.01 * exp(-0.5 * sum((X1[newaxis, :, :] - X2[:, newaxis, :])**2, axis=2)),
             x=X1,
             size=10)

Y2 = GP_draw(kern_func=lambda X1, X2: 0.01 * exp(-0.5 * sum((X1[newaxis, :, :] - X2[:, newaxis, :])**2, axis=2)),
             x=X2,
             size=10)

df = open("coupled.pkl","wb")
pickle.dump({
        "latent 1": X1,
        "latent 2": X2,
        "observed 1": Y1,
        "observed 2": Y2,
        "explanation": "trajectories [timesteps, dims] for latent spaces and observed spaces. with coupling between latent spaces"},
        df)
df.close()


plt.subplot(2,2,1)
plt.title("Phase diagram, latent space 1")
plt.plot(X1[:,0],X1[:,1])

plt.subplot(2,2,2)
plt.title("Phase diagram, latent space 2")
plt.plot(X2[:,0],X2[:,1])

plt.subplot(2,2,3)
plt.title("observed space 1")
plt.plot(arange(len(Y1)) * dt,Y1)

plt.subplot(2,2,4)
plt.title("observed space 2")
plt.plot(arange(len(Y2)) * dt,Y2)
         
plt.savefig("./synthetic/training.pdf")
plt.close()

titlesize = 20

fig = plt.figure(figsize=(5, 5))
plt.title("Latent space 1", fontsize=titlesize)
plt.plot(X1[:,0], X1[:,1], "k", linewidth=0.5)
plt.savefig("./synthetic/training-latent1.pdf")
plt.close()

fig = plt.figure(figsize=(5, 5))
plt.title("Latent space 2", fontsize=titlesize)
plt.plot(X2[:,0], X2[:,1], "k", linewidth=0.5)
plt.savefig("./synthetic/training-latent2.pdf")
plt.close()

fig = plt.figure(figsize=(5, 5))
plt.title("Observed space 1", fontsize=titlesize)
plt.plot(Y1[:300, :], linewidth=0.5)
plt.savefig("./synthetic/training-observed1-300.pdf")
plt.close()

fig = plt.figure(figsize=(5, 5))
plt.title("Observed space 2", fontsize=titlesize)
plt.plot(Y2[:300, :], linewidth=0.5)
plt.savefig("./synthetic/training-observed2-300.pdf")
plt.close()

                            
