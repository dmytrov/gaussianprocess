import numpy as np
import matplotlibex as plx
import ml.gptheano.vcgpdm.model as mdl
import numerical.numpytheano as nt
import matplotlibex as plx
import pickle


df = open("mocap/turn_right01..leftArm.npy", "rb")
left_arm = np.load(df)
df.close()

np.random.seed(250912)

X1 = pickled["latent 1"]
X2 = pickled["latent 2"]
Y1 = pickled["observed 1"]
Y2 = pickled["observed 2"]
Y = np.hstack((Y1, Y2))
#Y += 0.005 * np.reshape(np.random.normal(size=Y.size), Y.shape)
M = 55 # 50 # 27
X = [X1 - np.mean(X1, axis=0),
     X2 - np.mean(X2, axis=0),]
init_func = None
#init_func = lambda y, q, id: X[id]

print("|=========== NumPy ==========|")
ns = nt.NumpyLinalg
data = mdl.ModelData(Y, ns=ns)
params = mdl.ModelParams(data, Qs=[2, 2], parts_IDs=[0]*Y1.shape[1] + [1]*Y2.shape[1], M=M, init_func=init_func, ns=ns)
model = mdl.VCGPDM(params, ns=ns)

model.precalc_posterior_predictive()
x_pathes, y_pathes = model.run_generative_mode(100)
plx.plot_sequence2d(x_pathes[0])
plx.plot_sequences(y_pathes[0])

print("|=========== Theano ==========|")
ns = nt.TheanoLinalg
data = mdl.ModelData(Y, ns=ns)
params = mdl.ModelParams(data, Qs=[2, 2], parts_IDs=[0]*Y1.shape[1] + [1]*Y2.shape[1], M=M, init_func=init_func, ns=ns)
model = mdl.VCGPDM(params, ns=ns)
print("Coupling matrix: ", model.get_coupling_matrix_vales())        


print("|================================|")
model.optimize_kernel_params(maxiter=10)
#model.optimize_kernel_params(maxiter=10)
print("Coupling matrix: ", model.get_coupling_matrix_vales())        
model.precalc_posterior_predictive()
x_pathes, y_pathes = model.run_generative_mode(100)
plx.plot_sequence2d(x_pathes[0])
plx.plot_sequences(y_pathes[0])

print("|================================|")
model.optimize_all(maxiter=100)
print("Coupling matrix: ", model.get_coupling_matrix_vales())        
model.precalc_posterior_predictive()
x_pathes, y_pathes = model.run_generative_mode(100)
plx.plot_sequence2d(x_pathes[0])
plx.plot_sequences(y_pathes[0])

