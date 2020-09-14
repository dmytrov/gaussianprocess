import numpy as np
import os

def simple_scan(fn, 
                sequences=None,
                outputs_info=None,
                non_sequences=None,
                n_steps=None):
    """
    Basic implementation of theano.scan for numpy.
    Does not support taps.
    """
    def wrap_into_list(x):
        '''
        Wrap the input into a list if it is not already a list
        '''
        if x is None:
            return []
        elif not isinstance(x, (list, tuple)):
            return [x]
        else:
            return list(x)

    def all_equal(lst):
        return lst[1:] == lst[:-1]

    seqs = wrap_into_list(sequences)
    outs_info = wrap_into_list(outputs_info)

    non_seqs = []
    for elem in wrap_into_list(non_sequences):
        if not isinstance(elem, np.ndarray):
            non_seqs.append(np.asarray(elem))
        else:
            non_seqs.append(elem)

    seqs_sizes = [s.shape[0] for s in seqs]
    if n_steps is None:
        # sequences dim 0 must be same size
        assert len(seqs_sizes) >= 0
        assert all_equal(seqs_sizes)
        n_steps = seqs_sizes[0]
    else:
        # sequences dim 0 must be >= n_steps
        for s_size in seqs_sizes:
            assert s_size >= n_steps

    outputs_info_shape = outputs_info.shape
    outputs = np.concatenate([outputs_info[np.newaxis,...], np.full([n_steps] + list(outputs_info_shape), np.NaN, dtype=outputs_info.dtype)])
    for step in range(n_steps):
        arg_sequences = [s[step] for s in seqs]
        arg_outputs = [outputs[step]]
        args = arg_sequences + arg_outputs + non_seqs
        output = fn(*args)
        assert output.shape == outputs_info_shape
        outputs[step+1] = output

    return outputs[1:, ...], None


# Export as scan
scan = simple_scan


if __name__ == "__main__":
    A = np.array([[1, 2], [3, 4]])
    k = 10
    result = 1
    for i in range(k):
        result = result * A
    print("Direct numpy loop:")
    print(result)
    res_np = result

    result = simple_scan(
        fn=lambda prior_result, A: prior_result * A,
        outputs_info=np.ones_like(A),
        non_sequences=A,
        n_steps=k)
    res_scan = result[-1]
    print("Using numpy scan:")
    print(res_scan)
    print("numpy scan shape:")
    print(result.shape)
    
    assert np.all(res_np == res_scan)