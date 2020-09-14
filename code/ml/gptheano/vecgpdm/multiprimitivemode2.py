
"""New implementation of the multiprimitive model. 
v0.2
"""



class VarGPLVM(object):
    """
    LVM with variational treatment
    """
    def __init__(self, y, kern_class):
        self.y = y
        self.post_x_mean = None
        self.post_x_covar = None
        self.aug_in = None
        self.aug_out = None
        self.kern_class = kern_class



class VarCGPDM(object):
    """
    Dynamics prior with variational inference
    """
    def __init__(self, post_x_means, post_x_covars, couplings, kern_class):
        self.post_x_means = post_x_means
        self.post_x_covars = post_x_covars
        self.aug_in = None
        self.aug_out = None
        self.kern_class = kern_class
    
        


class PartType(object):
    def __init__(self, channels):
        self.channels = channels
    
        self.approx_gp = None  # ApproxGP()
        

    def init_latents(self):
        pass



class MPType(object):
    def __init__(self, name):
        self.name = name

        self.aug_ins = None
    


class Trial(object):
    def __init__(self, y_data):
        self.y_data = y_data



class Segment(object):
    def __init__(self, part_type, mp_type, start, stop, trial):
        self.trial, self.part_type, self.mp_type = trial, part_type, mp_type
        assert start < stop
        self.start, self.stop = start, stop

        self.previous = None  # the very previous segment for topoligical connectivity



class MotionGroup(object):
    def __init__(self):
        self.mp = {}  # part_type->mp_type dict
        self.segments = []


    def construct_elbo(self):
        pass


    def construct_likelihood(self):
        pass



class Dataset(object):
    def __init__(self):
        self.trials = []  # list of trails
        self.segments = []  # list of segments
        self.groups = []  # list of groups


    def add_trial(self, trial):
        self.trials.append(trial)
        return trial


    def add_segment(self, segment):
        if segment.trial is None:
            segment.trial == self.trials[-1]
        self.segments.append(segment)
        return segment


    def group_segments(self):
        pass