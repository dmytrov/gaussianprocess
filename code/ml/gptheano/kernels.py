import functools as ft
import numpy as np
import numerical.numpytheano as nt
import ml.gp.equations as eq
from numerical.theanoext.parametricfunction import *


class MultiSequenceKernel(ParametricFunction):
    def set_sequences_indexes(self, X1sequencesindexes, X2sequencesindexes):
        for child in self._children:
            child.set_sequences_indexes(X1sequencesindexes, X2sequencesindexes)

class RBFKernel(ParametricFunction):
    def __init__(self, X1Var, X2Var, LVar=None, SVar=None):
        super(RBFKernel, self).__init__()
        if LVar is None:
            LVar = ScalarVariable("RBF.L", 1.0, (-10.0, 10.0))
        if SVar is None:
            SVar = ScalarVariable("RBF.S", 1.0, (-10.0, 10.0))
        self.X1Var, self.X2Var, self.LVar, self.SVar = X1Var, X2Var, LVar, SVar
        self._args = [self.X1Var, self.X2Var]
        self._params = [self.LVar, self.SVar]
        self.symbolic = eq.RBF_kernel(self.X1Var.symbolic, self.X2Var.symbolic, self.LVar.symbolic, self.SVar.symbolic, ns=nt.TheanoLinalg)
        self.function = lambda X1, X2: eq.RBF_kernel(X1, X2, self.LVar.val, self.SVar.val, ns=nt.NumpyLinalg)

class NoiseKernel(ParametricFunction):
    def __init__(self, X1Var, X2Var, NVar=None):
        super(NoiseKernel, self).__init__()
        if NVar is None:
            NVar = ScalarVariable("Noise.N", 1.0, (-2.0, 2.0))
        self.X1Var, self.X2Var, self.NVar = X1Var, X2Var, NVar
        self._args = [self.X1Var, self.X2Var]
        self._params = [self.NVar]
        self.symbolic = eq.noise_kernel(self.X1Var.symbolic, self.X2Var.symbolic, self.NVar.symbolic, ns=nt.TheanoLinalg)
        self.function = lambda X1, X2: eq.noise_kernel(X1, X2, self.NVar.val, ns=nt.NumpyLinalg)

class RBFSecondOrderKernel(ParametricFunction):
    def __init__(self, X1Var, X2Var, LVar=None, S1Var=None, S2Var=None):
        super(RBFSecondOrderKernel, self).__init__()
        if LVar is None:
            LVar = ScalarVariable("RBF2.L", 1.0, (-10.0, 10.0))
        if S1Var is None:
            S1Var = ScalarVariable("RBF2.S1", 1.0, (-10.0, 10.0))
        if S2Var is None:
            S2Var = ScalarVariable("RBF2.S2", 1.0, (-10.0, 10.0))
        self.X1Var, self.X2Var, self.LVar, self.S1Var, self.S2Var = X1Var, X2Var, LVar, S1Var, S2Var
        self._args = [self.X1Var, self.X2Var]
        self._params = [self.LVar, self.S1Var, self.S2Var]
        self.symbolic = eq.RBF_2order_kernel(self.X1Var.symbolic, self.X2Var.symbolic,
                                               self.LVar.symbolic,
                                               self.S1Var.symbolic, self.S2Var.symbolic, ns=nt.TheanoLinalg)
        self.function = lambda X1, X2: eq.RBF_2order_kernel(X1, X2, self.LVar.val, self.S1Var.val, self.S2Var.val, ns=nt.NumpyLinalg)

class NoiseSecondOrderKernel(ParametricFunction):
    def __init__(self, X1Var, X2Var, NVar=None):
        super(NoiseSecondOrderKernel, self).__init__()
        if NVar is None:
            NVar = ScalarVariable("Noise2.N", 1.0, (-2.0, 2.0))
        self.X1Var, self.X2Var, self.NVar = X1Var, X2Var, NVar
        self._args = [self.X1Var, self.X2Var]
        self._params = [self.NVar]
        self.symbolic = eq.noise_2order_kernel(self.X1Var.symbolic, self.X2Var.symbolic, self.NVar.symbolic, ns=nt.TheanoLinalg)
        self.function = lambda X1, X2: eq.noise_2order_kernel(X1, X2, self.NVar.val, ns=nt.NumpyLinalg)

class SumKernel(ParametricFunction):
    def __init__(self, children=[]):
        super(SumKernel, self).__init__()
        self._children = children
        self.symbolic = sum([child.symbolic for child in children])
        self.function = lambda X1, X2: ft.reduce(np.add, [child.function(X1, X2) for child in children])

class SumMultiSequenceKernel(MultiSequenceKernel):
    def __init__(self, children=[]):
        super(SumMultiSequenceKernel, self).__init__()
        self._children = children
        self.symbolic = sum([child.symbolic for child in children])
        self.function = lambda X1, X2, X1sequencesindexes, X2sequencesindexes: \
            ft.reduce(np.add, [child.function(X1, X2, X1sequencesindexes, X2sequencesindexes) for child in children])

class RBFSecondOrderMultiSequenceKernel(MultiSequenceKernel):
    def __init__(self, X1Var, X2Var, LVar=None, S1Var=None, S2Var=None, X1tplusIndexesVar=None, X2tplusIndexesVar=None):
        super(RBFSecondOrderMultiSequenceKernel, self).__init__()
        if LVar is None:
            LVar = ScalarVariable("RBF2.L", 1.0, (-10.0, 10.0))
        if S1Var is None:
            S1Var = ScalarVariable("RBF2.S1", 1.0, (-10.0, 10.0))
        if S2Var is None:
            S2Var = ScalarVariable("RBF2.S2", 1.0, (-10.0, 10.0))
        if X1tplusIndexesVar is None:
            X1tplusIndexesVar = IntVectorVariable("RBF2.X1tplusIndexes")
        if X2tplusIndexesVar is None:
            X2tplusIndexesVar = IntVectorVariable("RBF2.X2tplusIndexes")
        self.X1Var, self.X2Var, self.LVar, self.S1Var, self.S2Var, self.X1tplusIndexesVar, self.X2tplusIndexesVar = \
            X1Var, X2Var, LVar, S1Var, S2Var, X1tplusIndexesVar, X2tplusIndexesVar
        self._args = [self.X1Var, self.X2Var]
        self._params = [self.LVar, self.S1Var, self.S2Var]
        self._consts = [self.X1tplusIndexesVar, self.X2tplusIndexesVar]
        self.symbolic = eq.RBF_2order_multi_sequence_kernel(self.X1Var.symbolic, self.X2Var.symbolic,
                                                              self.X1tplusIndexesVar.symbolic,
                                                              self.X2tplusIndexesVar.symbolic,
                                                              self.LVar.symbolic,
                                                              self.S1Var.symbolic,
                                                              self.S2Var.symbolic,
                                                              ns=nt.TheanoLinalg)
        self.function = lambda X1, X2, X1tplusIndexes, X2tplusIndexes: eq.RBF_2order_multi_sequence_kernel(X1, X2,
                                                                           X1tplusIndexes,
                                                                           X2tplusIndexes,
                                                                           self.LVar.val,
                                                                           self.S1Var.val,
                                                                           self.S2Var.val,
                                                                           ns=nt.NumpyLinalg)

    def set_sequences_indexes(self, X1sequencesindexes, X2sequencesindexes):
        super(RBFSecondOrderMultiSequenceKernel, self).set_sequences_indexes(X1sequencesindexes, X2sequencesindexes)
        self.X1tplusIndexesVar.val = np.hstack([np.array(sequenceindexes[2:]) for sequenceindexes in X1sequencesindexes])
        self.X2tplusIndexesVar.val = np.hstack([np.array(sequenceindexes[2:]) for sequenceindexes in X2sequencesindexes])

class NoiseSecondOrderMultiSequenceKernel(MultiSequenceKernel):
    def __init__(self, X1Var, X2Var, NVar=None, X1tplusIndexesVar=None, X2tplusIndexesVar=None):
        super(NoiseSecondOrderMultiSequenceKernel, self).__init__()
        if NVar is None:
            NVar = ScalarVariable("Noise2.N", 1.0, (-2.0, 2.0))
        if X1tplusIndexesVar is None:
            X1tplusIndexesVar = IntVectorVariable("Noise2.X1tplusIndexes")
        if X2tplusIndexesVar is None:
            X2tplusIndexesVar = IntVectorVariable("Noise2.X2tplusIndexes")
        self.X1Var, self.X2Var, self.NVar, self.X1tplusIndexesVar, self.X2tplusIndexesVar = \
            X1Var, X2Var, NVar, X1tplusIndexesVar, X2tplusIndexesVar
        self._args = [self.X1Var, self.X2Var]
        self._params = [self.NVar]
        self._consts = [self.X1tplusIndexesVar]  #, self.X2tplusIndexesVar]
        self.symbolic = eq.noise_2order_multi_sequence_kernel(self.X1Var.symbolic, self.X2Var.symbolic,
                                                                self.X1tplusIndexesVar.symbolic, self.X2tplusIndexesVar.symbolic,
                                                                self.NVar.symbolic, ns=nt.TheanoLinalg)
        self.function = lambda X1, X2, X1tplusIndexes, X2tplusIndexes: \
            eq.noise_2order_multi_sequence_kernel(X1, X2, X1tplusIndexes, X2tplusIndexes,
                                                  self.NVar.val, ns=nt.NumpyLinalg)

    def set_sequences_indexes(self, X1sequencesindexes, X2sequencesindexes):
        super(NoiseSecondOrderMultiSequenceKernel, self).set_sequences_indexes(X1sequencesindexes, X2sequencesindexes)
        self.X1tplusIndexesVar.val = np.hstack([np.array(sequenceindexes[2:]) for sequenceindexes in X1sequencesindexes])
        self.X2tplusIndexesVar.val = np.hstack([np.array(sequenceindexes[2:]) for sequenceindexes in X2sequencesindexes])

#######################################################
#  Generic multi-dimensional tensor kernels
#######################################################

class SumKern(ParametricFunction):
    def __init__(self, children=[]):
        super(SumKern, self).__init__()
        self._children = children
        self.symbolic = sum([child.symbolic for child in children])
        self.function = lambda X1, X2: ft.reduce(np.add, [child.function(X1, X2) for child in children])


class RowMap(ParametricFunction):
    def __init__(self, Var, indexes=np.arange(2)):
        super(RowMap, self).__init__()
        self.IndexesVar = IntVectorVariable("RowIndexes", indexes)
        self._consts = [self.IndexesVar]
        self.symbolic = Var.symbolic[self.IndexesVar.symbolic]
        self.function = lambda X: X[self.IndexesVar.val]


class ColumnMap(ParametricFunction):
    def __init__(self, Var, indexes=np.arange(2)):
        super(ColumnMap, self).__init__()
        self.IndexesVar = IntVectorVariable("ColumnIndexes", indexes)
        self._consts = [self.IndexesVar]
        self.symbolic = Var.symbolic[:, self.IndexesVar.symbolic]
        self.function = lambda X: X[:, self.IndexesVar.val]


def XtoXtminusMap(XVar, xtminusindexes=[np.arange(2), np.arange(3)]):
    return [RowMap(XVar, indexes) for indexes in xtminusindexes]


class RBFKern(ParametricFunction):
    def __init__(self, X1Vars, X2Vars=None, LVar=None, SVars=[None]):
        super(RBFKern, self).__init__()
        self.dynamicsOrder = len(X1Vars)
        if LVar is None:
            LVar = ScalarVariable("RBF.L", 1.0, (-10.0, 10.0))
        if SVars[0] is None:
            SVars = [ScalarVariable("RBF.S"+str(i), 1.0, (-10.0, 10.0)) for i in range(self.dynamicsOrder)]
        self.X1Vars, self.X2Vars, self.LVar, self.SVars = X1Vars, X2Vars, LVar, SVars
        self._args = []
        self._params = [self.LVar] + self.SVars
        self.symbolic = eq.RBF_kern([x1.symbolic for x1 in self.X1Vars],
                                    None if self.X2Vars is None else [x2.symbolic for x2 in self.X2Vars],
                                    self.LVar.symbolic,
                                    [s.symbolic for s in self.SVars],
                                    ns=nt.TheanoLinalg)
        self.function = lambda X1, X2: eq.RBF_kern(X1, X2, self.LVar.val,
                                                            [s.val for s in self.SVars],
                                                            ns=nt.NumpyLinalg)


class LinearKern(ParametricFunction):
    def __init__(self, X1Vars, X2Vars=None, SVars=[None]):
        super(LinearKern, self).__init__()
        self.dynamicsOrder = len(X1Vars)
        if SVars[0] is None:
            SVars = [ScalarVariable("Linear.S"+str(i), 1.0, (-10.0, 10.0)) for i in range(self.dynamicsOrder)]
        self.X1Vars, self.X2Vars, self.SVars = X1Vars, X2Vars, SVars
        self._args = []
        self._params = self.SVars
        self.symbolic = eq.linear_kern([x1.symbolic for x1 in self.X1Vars],
                                    None if self.X2Vars is None else [x2.symbolic for x2 in self.X2Vars],
                                    [s.symbolic for s in self.SVars],
                                    ns=nt.TheanoLinalg)
        self.function = lambda X1, X2: eq.linear_kern(X1, X2, [s.val for s in self.SVars], ns=nt.NumpyLinalg)


class NoiseKern(ParametricFunction):
    def __init__(self, X1Vars, X2Vars=None, NVar=None):
        super(NoiseKern, self).__init__()
        if NVar is None:
            NVar = ScalarVariable("Noise.N", 1.0, (-2.0, 2.0))
        self.X1Vars, self.X2Vars, self.NVar = X1Vars, X2Vars, NVar
        self._args = []
        self._params = [self.NVar]
        if self.X1Vars is self.X2Vars:
            self._params = [self.NVar]
            x1 = [x1.symbolic for x1 in self.X1Vars]
            x2 = x1
        else:
            x1 = [x1.symbolic for x1 in self.X1Vars]
            x2 = None if self.X2Vars is None else [x2.symbolic for x2 in self.X2Vars]
        self.symbolic = eq.noise_kern(x1, x2, self.NVar.symbolic, ns=nt.TheanoLinalg)
        self.function = lambda X1, X2: eq.noise_kern(X1, X2, self.NVar.val, ns=nt.NumpyLinalg)


class CouplingKern(ParametricFunction):
    def __init__(self, partial_kerns_vars, log_sigma_sqrs_var=None):
        super(CouplingKern, self).__init__()
        self.partial_kerns = partial_kerns_vars
        n_kerns = len(partial_kerns_vars)
        if log_sigma_sqrs_var is None:
            log_sigma_sqrs_var = MatrixVariable("Coupling.Sigmas", np.ones([n_kerns, n_kerns]), (-2.0, 2.0))
        self.log_sigma_sqrs_var = log_sigma_sqrs_var
        self._params.append(self.log_sigma_sqrs_var)
        self._children.extend(partial_kerns_vars)
        self.symbolic = eq.couple_kerns([kern.symbolic for kern in self.partial_kerns],
                                        self.log_sigma_sqrs_var.symbolic,
                                        X1_is_X2=True,
                                        ns=nt.TheanoLinalg)
        self.function = lambda partial_kerns, X1_is_X2: eq.couple_kerns(partial_kerns,
                                                                        self.log_sigma_sqrs_var.val,
                                                                        X1_is_X2=X1_is_X2,
                                                                        ns=nt.NumpyLinalg)



























