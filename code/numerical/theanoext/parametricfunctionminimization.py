from scipy import optimize
import numerical.theanoext.parametricfunction as pf

def cg(functominimize, vars=[], maxiter=100):
    vars = pf.unique(vars)
    print("Optimizing ", pf.symbols(vars))
    functominimize.set_wrts(pf.symbols(vars))
    xOpt = optimize.fmin_cg(f=functominimize.get_func_value_wrt_vec,
                            x0=pf.get_values_vec(vars),
                            fprime=functominimize.grad_vec,
                            disp=1, maxiter=maxiter)
    pf.set_values_vec(vars, xOpt)

def ncg(functominimize, vars=[], maxiter=100):
    vars = pf.unique(vars)
    print("Optimizing ", pf.symbols(vars))
    functominimize.set_wrts(pf.symbols(vars))
    xOpt = optimize.fmin_ncg(f=functominimize.get_func_value_wrt_vec,
                                   x0=pf.get_values_vec(vars),
                                   fprime=functominimize.grad_vec,
                                   disp=1, maxiter=maxiter)
    pf.set_values_vec(vars, xOpt)

def l_bfgs_b(functominimize, vars=[], maxiter=100):
    vars = pf.unique(vars)
    print("Optimizing ", pf.symbols(vars))
    functominimize.set_wrts(pf.symbols(vars))
    xOpt, f, d = optimize.fmin_l_bfgs_b(functominimize.get_func_value_and_grad_vec,
                                       x0=pf.get_values_vec(vars),
                                       bounds=pf.get_bounds(vars),
                                       disp=1, maxiter=maxiter)
    pf.set_values_vec(vars, xOpt)
