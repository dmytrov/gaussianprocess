### Test data generator for vCGPDM learning
from numpy import *
import matplotlib.pyplot as plt
import itertools,pickle

class GPFunc:
    """Gaussian process function generator"""

    def __init__(self,inducingInputs,meanFunction,kernelFunction):
        """Draw the function
        inducingInputs[M,D] with M: number of inducing points, D: input dimension
        meanFunction(X), kernelFunction(X,X') for X \in R^D"""

        self.X=ndarray((len(inducingInputs),),dtype="object") # transform into object array so we can use numpy's 'outer' ufunc member
        for i in range(len(inducingInputs)): self.X[i]=inducingInputs[i]
    
        self.kernelFunction=lambda X1,X2:frompyfunc(kernelFunction,2,1).outer(X1,X2).astype("float64")
        self.meanFunction=lambda X:frompyfunc(meanFunction,1,1)(X).astype("float64")
        # draw inducing function values
        self.Kmm=self.kernelFunction(self.X,self.X)
        self.mum=self.meanFunction(self.X)
        self.F=random.multivariate_normal(self.mum,self.Kmm)
        self.Kmmi=linalg.inv(self.Kmm)
        
        self.Kmmi=linalg.inv(self.kernelFunction(self.X,self.X))

    def __call__(self,Xtest):
        """Given Xtest[N,D], return function values"""
        Xtest=atleast_2d(Xtest)
        Xt=ndarray((len(Xtest),),dtype="object")
        for i in range(len(Xt)): Xt[i]=Xtest[i]
        Knm=self.kernelFunction(Xt,self.X)
        return einsum("nm,mp,p->n",Knm,self.Kmmi,(self.F-self.mum))+self.meanFunction(Xt)


    
random.seed(250912)

TIMESTEPS=1000
dt=0.01 # time discretization
# the model has 2 latent spaces, 2D each

w1=2*pi*1.0 # base frequency of oscillation in first latent space
delta1=0.3 # dampening 
inducingPoints=vstack([random.uniform(-5.0,5.0,size=100),random.uniform(-5*w1,5*w1,size=100)]).T
latentMapping11=GPFunc(inducingPoints,lambda X:X[0]+X[1]*dt,lambda X1,X2:0.05*exp(-1*linalg.norm(X1-X2)**2))
latentMapping12=GPFunc(inducingPoints,lambda X:X[1]-(w1**2*X[0]+2.0*delta1*w1*X[1])*dt,lambda X1,X2:0.1*exp(-3.0/w1**2*linalg.norm(X1-X2)**2))

w2=2*pi*0.7 # base frequency of oscillation in second latent space
delta2=0.4 # dampening 
inducingPoints=vstack([random.uniform(-5.0,5.0,size=100),random.uniform(-5*w2,5*w2,size=100)]).T
latentMapping21=GPFunc(inducingPoints,lambda X:X[0]+X[1]*dt,lambda X1,X2:0.01*exp(-0.5*linalg.norm(X1-X2)**2))
latentMapping22=GPFunc(inducingPoints,lambda X:X[1]-(w2**2*X[0]+2.0*delta2*w2*X[1])*dt,lambda X1,X2:0.01*exp(-4.0/w2**2*linalg.norm(X1-X2)**2))

# create time series in latent spaces
X1=[array((1.0,0.0))]
X2=[array((0.0,-w2))]
for t in arange(1,TIMESTEPS):
    nextX1=hstack([latentMapping11(X1[-1]),latentMapping12(X1[-1])])
    nextX2=hstack([latentMapping21(X2[-1]),latentMapping22(X2[-1])]) *0.9+nextX1*0.1
    X1+=[nextX1]
    X2+=[nextX2]

X1=array(X1)
X2=array(X2)


# latent-to-observed mappings
inducingPoints=vstack([random.uniform(-5.0,5.0,size=100),random.uniform(-5*w1,5*w1,size=100)]).T
latentToObserved1=GPFunc(inducingPoints,lambda X:X[0]*2.0+0.3,lambda X1,X2:0.01*exp(-0.5*linalg.norm(X1-X2)**2))
latentToObserved2=GPFunc(inducingPoints,lambda X:X[0]**3,lambda X1,X2:0.01*exp(-0.5*linalg.norm(X1-X2)**2))

Y1=latentToObserved1(X1)
Y2=latentToObserved2(X2)


plt.subplot(2,2,1)
plt.title("Phase diagram, latent space 1")
plt.plot(X1[:,0],X1[:,1])

plt.subplot(2,2,2)
plt.title("Phase diagram, latent space 2")
plt.plot(X2[:,0],X2[:,1])

plt.subplot(2,2,3)
plt.title("observed space 1")
plt.plot(arange(len(Y1))*dt,Y1)

plt.subplot(2,2,4)
plt.title("observed space 2")
plt.plot(arange(len(Y2))*dt,Y2)

df=open("coupled.pkl","wb")
pickle.dump({"latent 1":X1,"latent 2":X2,"observed 1":Y1,"observed 2":Y2,"explanation":"trajectories [timesteps,dims] for latent spaces and observed spaces. with coupling between latent spaces"},df)
df.close()

         
plt.show()
                            
