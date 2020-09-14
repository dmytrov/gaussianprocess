from __future__ import print_function
import types
import functools as ft


class PrePostCaller:
    """
    Wraps an object
    """

    def __init__(self, other):
        self.other = other

    def pre(self):
        print('pre')

    def post(self):
        print('post')

    def __getattr__(self, name):
        if hasattr(self.other, name):
            func = getattr(self.other, name)
            return lambda *args, **kwargs: self._wrap(func, args, kwargs)
        raise AttributeError(name)

    def _wrap(self, func, args, kwargs):
        self.pre()
        if type(func) == types.MethodType:
            result = func(*args, **kwargs)
        else:
            result = func(self.other, *args, **kwargs)
        self.post()
        return result


class CurriedFunction(object):
    """
    The same as functools.partial
    """

    def __init__(self, func, *fixedargs):
        self.func = func
        self.fixedargs = fixedargs

    def __call__(self, *args, **kwargs):
        return self.func(*(self.fixedargs + args), **kwargs)


class FunctionWrapper(object):
    """
    Similar to functools.update_wrapper but calls the inner function too
    """

    def __init__(self, inner, outer):
        self.inner = inner
        self.outer = outer

    def __call__(self, *args, **kwargs):
        self.outer(*args, **kwargs)
        return self.inner(*args, **kwargs)


def wrap_function(inner, outer):
    return FunctionWrapper(inner, outer)


def wrap_functions(inners, outers):
    return [wrap_function(inner, outer) for inner, outer in zip(inners, outers)]


def wrap_method(inner, outer):
    """
    Wraps a method.
    Calls innter method after the outer
    """
    obj = inner.im_self

    def caller(*args, **kwargs):
        outer(*args, **kwargs)
        return inner(*args[1:], **kwargs)
    caller.__name__ = inner.__name__
    wrapper = types.MethodType(caller, obj, type(obj))
    setattr(obj, inner.__name__, wrapper)


def override_method(inner, outer):
    """
    Wraps a method.
    Calls outer method only
    """
    obj = inner.im_self

    def caller(*args, **kwargs):
        return outer(inner, *args, **kwargs)
    caller.__name__ = inner.__name__
    wrapper = types.MethodType(caller, obj, type(obj))
    setattr(obj, inner.__name__, wrapper)


# def __wrap_methods(inners, outers, inplace=True):
#     """
#     Wraps object methods.
#     Creates a new class type dynamically or replaces existing methods inplace.
#     inners : list of methods to be wrapped
#     outers : list of wrappers
#     inplace : create a new class type dynamically or replaces existing methods inplace
#     Returns wrapped object
#     """
#     try:
#         obj = inners[0].im_self
#     except AttributeError as err:
#         obj = inners[0].__self__
#     obj = inners[0].im_self
#     if inplace:
#         res = obj
#     else:
#         res = type(type(obj).__name__ + "_W", (type(obj),), {})()
#     for inner, outer in zip(inners, outers):
#         assert inner.im_self == obj
#         wrapper = FunctionWrapper(inner, CurriedFunction(outer, obj))
#         setattr(res, inner.__name__, wrapper.__call__)
#     return res


# def _wrap_methods(inners, outers, inplace=False):
#     """
#     Wraps object methods.
#     Creates a new class type dynamically or replaces existing methods inplace.
#     inners : list of methods to be wrapped
#     outers : list of wrappers
#     inplace : create a new class type dynamically or replaces existing methods inplace
#     Returns wrapped object
#     """
#     obj = inners[0].im_self
#     if inplace:
#         res = obj
#     else:
#         res = type(type(obj).__name__ + "_W", (type(obj),), {})()
#     for inner, outer in zip(inners, outers):
#         assert inner.im_self == obj
#         wrapper = ft.update_wrapper(ft.partial(outer, obj), inner)
#         setattr(res, inner.__name__, wrapper)
#     return res


if __name__ == "__main__":

    class Printer(object):

        def __init__(self, x):
            self.x = x

        def sayA(self, a):
            print("printerA" + a)

        def sayB(self, a):
            print("printerB" + a)

    class Filter(object):

        def __init__(self, x):
            self.x = x

        def sayA(self, other, a):
            print("filterA" + a)

        def sayB(self, inner, other, a):
            print("filterB" + a)
            return inner(a)

    def printwrapperA(self, a):
        print("printwrapperA" + a)

    def printwrapperB(self, a):
        print("printwrapperB" + a)

    def printwrapperC(self, a):
        print("printwrapperC" + a)

    def printwrapperfunc(a):
        print("printwrapper" + a)

    p = Printer(1)
    wrap_method(p.sayA, printwrapperA)
    wrap_method(p.sayA, printwrapperB)
    wrap_method(p.sayA, printwrapperC)
    f = Filter(2)
    wrap_method(p.sayA, f.sayA)
    override_method(p.sayB, f.sayB)
    p.sayA("1")
    p.sayB("2")


    wrapped = wrap_function(printwrapperA, printwrapperB)
    #wrapped(None, "3")
