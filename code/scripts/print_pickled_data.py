from __future__ import print_function
import sys
from six.moves import cPickle

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
        if len(sys.argv) >= 3:
            substr = sys.argv[2]
        else:
            substr = None
        print("Reading state from {}".format(filename))
        with open(filename, "rb") as filehandle:
            vars_state = cPickle.load(filehandle)
        for k,v in vars_state.iteritems():
            if substr is None or substr in k:
                print(k, v)
    else:
        print("Usage: " + __file__ + " filename.pkl [filter]")