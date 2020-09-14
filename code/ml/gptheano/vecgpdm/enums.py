
class VarTag(object):
    latent_x = 0
    kernel_params = 1
    couplings = 2
    augmenting_inputs = 3
    observed = 4
    data_mean = 5

class EstimationMode(object):
    ELBO = 0
    MAP = 1
