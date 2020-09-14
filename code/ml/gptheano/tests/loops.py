import numpy as np
import theano
import theano.tensor as T


samples = T.matrix("samples")
starts = T.vector("indexes", dtype="int32")


def joint(i1, prev_out, samples):
    return T.concatenate([prev_out, samples])

def fill(i, k1, k2, prev_out, samples):
    return samples[k2]
    #return T.concatenate([in_out[:i, :], samples[k, :], in_out[i+1:, :]])

filled, updates = theano.scan(fn=fill,
                     sequences=[T.arange(starts.shape[0]), dict(input=starts, taps=[-1, 0])],
                     outputs_info=T.zeros([samples.shape[1]]),
                     non_sequences=samples)
A = samples[starts]
makeXminus= theano.function([samples, starts], A)

print(makeXminus(np.identity(10), np.array([2, 4, 8])))

print("#####################")
import theano.typed_list

tl = theano.typed_list.TypedListType(T.ivector)
def ranges(k1, k2, k3):
    #theano.typed_list.append(tl, T.arange(k1, k2))
    return [k1, k2, k3]

B, _ = theano.scan(fn=ranges,
                   sequences=[T.arange(starts.shape[0]),
                              dict(input=starts, taps=[-1, 0])]
                   )

makeXminus= theano.function([starts], B)

print(makeXminus(np.array([2, 4, 8])))

