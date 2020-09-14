import numpy as np
import matplotlibex as plx
import ml.gptheano.vcgpdm.model as mdl
import numerical.numpytheano as nt
import matplotlibex as plx
import matplotlib.pyplot as plt
import pickle
import theano

print("Theano mode: ", theano.config.mode)

df = open("coupled.pkl", "rb")
pickled = pickle.load(df) 
df.close()

np.random.seed(250912)

X1 = pickled["latent 1"]
X2 = pickled["latent 2"]
Y1 = pickled["observed 1"]
Y2 = pickled["observed 2"]

sl = [slice(500, None),slice(None)]
X1 = X1[sl]
X2 = X2[sl]
Y1 = Y1[sl]
Y2 = Y2[sl]

# Inject common mean 
Y1[:, :Y1.shape[1]/2] += 0.5 * X1[:, 0][:, np.newaxis]
Y1[:, Y1.shape[1]/2:] += 0.5 * X1[:, 1][:, np.newaxis]
Y2[:, :Y1.shape[1]/2] += 0.5 * X2[:, 0][:, np.newaxis]
Y2[:, Y1.shape[1]/2:] += 0.5 * X2[:, 1][:, np.newaxis]

Y_1_2 = np.hstack((Y1, Y2))
Y_1_2 = Y_1_2 + 0.05 * np.reshape(np.random.normal(size=Y_1_2.size), Y_1_2.shape)
nsamples = 230
Y = Y_1_2[:nsamples, :]
Y_after_training = Y_1_2[1000:1200, :]
#for i in range(10): 
#    Y = np.vstack((Y, Y_1_2 + 0.0005 * np.reshape(np.random.normal(size=Y_1_2.size), Y_1_2.shape)))

M = (8, 16) # 50 # 27
X = [X1 - np.mean(X1, axis=0),
     X2 - np.mean(X2, axis=0),]
init_func = None
#init_func = lambda y, q, id: X[id]

print("|===================== NumPy =====================|")
ns = nt.NumpyLinalg
data = mdl.ModelData(Y, ns=ns)
params = mdl.ModelParams(data, Qs=[2, 2], parts_IDs=[0]*Y1.shape[1] + [1]*Y2.shape[1], M=M, init_func=init_func, ns=ns)
model = mdl.VCGPDM(params, ns=ns)
print("Variance explained training: ")
print(mdl.variance_explained_training(model))
print("Variance explained after-training data: ")
print(mdl.variance_explained_generated(model, after_training_data=Y_after_training))


print("|===================== Theano ====================|")
ns = nt.TheanoLinalg
data = mdl.ModelData(Y, ns=ns)
params = mdl.ModelParams(data, Qs=[2, 2], parts_IDs=[0]*Y1.shape[1] + [1]*Y2.shape[1], M=M, init_func=init_func, ns=ns)
model = mdl.VCGPDM(params, ns=ns)
model.print_min_max_x_covars_diags()
mdl.save_plots(model, nsteps=150, directory="./01")
mdl.plot_model(model)
mdl.plot_elbo_wrt_x_covars(model)

model.optimize_inducing(maxiter=1000)
mdl.plot_model(model)
mdl.plot_elbo_wrt_x_covars(model)

model.optimize_kernel_params(maxiter=10)
mdl.save_plots(model, nsteps=150, directory="./02")
mdl.plot_model(model)
mdl.plot_elbo_wrt_x_covars(model)

model.optimize_inducing(maxiter=1000)
mdl.plot_model(model)
mdl.plot_elbo_wrt_x_covars(model)

model.optimize_x(maxiter=1000)
mdl.save_plots(model, nsteps=150, directory="./03")
mdl.plot_model(model)
mdl.plot_elbo_wrt_x_covars(model)

model.optimize_inducing(maxiter=1000)
mdl.save_plots(model, nsteps=150, directory="./04")
mdl.plot_model(model)
mdl.plot_elbo_wrt_x_covars(model)

model.optimize_kernel_params(maxiter=100)
mdl.save_plots(model, nsteps=150, directory="./05")
mdl.plot_model(model)
mdl.plot_elbo_wrt_x_covars(model)

model.optimize_inducing(maxiter=1000)
mdl.plot_model(model)
mdl.plot_elbo_wrt_x_covars(model)

print("Variance explained training: ")
print(mdl.variance_explained_training(model))
print("Variance explained training: ")
print(mdl.variance_explained_generated(model, after_training_data=Y_after_training))

model.print_min_max_x_covars_diags()
print("|===================== Fixed x_covars_diags =====================|")
model.fix_x_covars_diags()
model.print_min_max_x_covars_diags()
print("neg-ELBO: ", model.functominimize.get_func_value())

model.optimize_kernel_params(maxiter=10)
mdl.plot_model(model)
mdl.plot_elbo_wrt_x_covars(model)

model.optimize_x(maxiter=1000)
mdl.plot_model(model)
mdl.plot_elbo_wrt_x_covars(model)


print("|===================== Setting all couplings equal to c[0, 0] ====================|")
model.log_couplingcovars_var.val = np.log(2 * np.exp(model.log_couplingcovars_var.val[0, 0])) * np.ones_like(model.log_couplingcovars_var.val)
mdl.plot_model(model)
model.optimize_kernel_params(maxiter=100)
mdl.save_plots(model, nsteps=150, directory="./06")
mdl.plot_model(model)

