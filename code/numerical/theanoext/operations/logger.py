import theano
import theano.tensor as T
import theano.tensor.nlinalg as nlinalg
import theano.gof as gof
import numpy as np
import numerical.numpyext.linalg as ntl


def screen_printer(value, message="#NONE_MESSAGE#"):
    print(message)
    print("Shape: {}".format(value.shape))
    print("Value: {}".format(value))


def matrix_viewer(value, message="#NONE_MESSAGE#"):
    import matplotlib.pyplot as plt 
    screen_printer(value, message)
    plt.imshow(value)
    plt.title(message)
    plt.show()


log_to_screen = T.printing.Print()
view_matrix = T.printing.Print(global_fn=lambda op, x: matrix_viewer(x))