import numpy as np
import collections

def vectorize(lst):
    if not isinstance(lst, list):
        return lst
        #lst = [lst]
    return np.hstack([np.ravel(val) for val in lst])
    
def unvectorizesingle(vec, template):
    if isinstance(template, float):
        return vec[0], vec[1:]
    elif isinstance(template, np.ndarray):
        return np.reshape(vec[:template.size], template.shape), vec[template.size:]
    else:
        raise Exception("Unsupported type: " + type(template).__name__)

def unvectorize(vec, templates):
    unvec = []
    for template in templates:
        u, vec = unvectorizesingle(vec, template)
        unvec.append(u)
    return unvec

