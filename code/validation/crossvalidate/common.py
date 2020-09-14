import numpy as np
import dataset.mocap as ds
import utils.parallel.runner as pr


def create_params(full=False):
    params = pr.Params()
    params.times("model", range=[
            "vcgpdm", 
            "dmp", 
            "tmp",
            ])
    params.times("dataset", range=["pass-bottle-hold"])
    params.times("mode", range=["ELBO", "MAP"], \
            when=lambda x: x["model"] in {"vcgpdm"})
    params.times("parts", range=[1, 3], \
            when=lambda x: x["model"] in {"vcgpdm", "cgpdm"})
    params.times("npsi", range=range(2, 51), \
            when=lambda x: x["model"] in {"dmp"})
    params.times("numprim", range=range(2, 11), \
            when=lambda x: x["model"] in {"tmp"})
    if full:
        r = range(2, 17, 1) + range(20, 41, 5)
    else:        
        r = range(4, 11, 1)
    params.times("dyn", range=r, \
            when=lambda x: x["model"] == "vcgpdm" and x["mode"] == "ELBO")
    params.times("lvm", range=r, \
            when=lambda x: x["model"] == "vcgpdm" and x["mode"] == "ELBO")
    params.times("hold", range=range(0, 5))
    return params



class ModelLerner(object):
    
    def __init__(self, dirname, param, args):
        self.dirname = dirname
        self.param = param
        self.args = args

    def load_data(self):
        if self.param["dataset"] == "walking":
            if "parts" not in self.param or self.param["parts"] == 1:
                bodypart_motiontypes=[(ds.BodyParts.FullBody, ds.MotionType.WALK_PHASE_ALIGNED)]
            elif self.param["parts"] == 2:
                bodypart_motiontypes=[
                        (ds.BodyParts.Upper, ds.MotionType.WALK_PHASE_ALIGNED),
                        (ds.BodyParts.Lower, ds.MotionType.WALK_PHASE_ALIGNED)]
            self.partitioner, self.parts_IDs, self.trials, self.starts_ends = \
                    ds.load_recording(
                    ds.Recordings.exp3_walk,
                    bodypart_motiontypes=bodypart_motiontypes,
                    max_chunks=5)
        elif self.param["dataset"] == "pass-bottle":
            if "parts" not in self.param or self.param["parts"] == 1:
                bodypart_motiontypes=[(ds.BodyParts.FullBody, ds.MotionType.PASS_BOTTLE_LEFT_TO_RIGHT)]
            elif self.param["parts"] == 3:
                bodypart_motiontypes=[
                        (ds.BodyParts.LeftArm, ds.MotionType.PASS_BOTTLE_LEFT_TO_RIGHT),
                        (ds.BodyParts.BodyNoArms, ds.MotionType.PASS_BOTTLE_LEFT_TO_RIGHT),
                        (ds.BodyParts.RightArm, ds.MotionType.PASS_BOTTLE_LEFT_TO_RIGHT)]
            self.partitioner, self.parts_IDs, self.trials, self.starts_ends = \
                    ds.load_recording(
                    ds.Recordings.exp4_pass_bottle_put,
                    bodypart_motiontypes=bodypart_motiontypes,
                    max_chunks=5)
        elif self.param["dataset"] == "return-bottle":
            if "parts" not in self.param or self.param["parts"] == 1:
                bodypart_motiontypes=[(ds.BodyParts.FullBody, ds.MotionType.PASS_BOTTLE_RIGHT_TO_LEFT)]
            elif self.param["parts"] == 3:
                bodypart_motiontypes=[
                        (ds.BodyParts.LeftArm, ds.MotionType.PASS_BOTTLE_RIGHT_TO_LEFT),
                        (ds.BodyParts.BodyNoArms, ds.MotionType.PASS_BOTTLE_RIGHT_TO_LEFT),
                        (ds.BodyParts.RightArm, ds.MotionType.PASS_BOTTLE_RIGHT_TO_LEFT)]
            self.partitioner, self.parts_IDs, self.trials, self.starts_ends = \
                    ds.load_recording(
                    ds.Recordings.exp4_pass_bottle_put,
                    bodypart_motiontypes=bodypart_motiontypes,
                    max_chunks=5)
        elif self.param["dataset"] == "pass-bottle-hold":
            if "parts" not in self.param or self.param["parts"] == 1:
                bodypart_motiontypes=[(ds.BodyParts.FullBody, ds.MotionType.PASS_BOTTLE_LEFT_TO_RIGHT)]
            elif self.param["parts"] == 3:
                bodypart_motiontypes=[
                        (ds.BodyParts.LeftArm, ds.MotionType.PASS_BOTTLE_LEFT_TO_RIGHT),
                        (ds.BodyParts.BodyNoArms, ds.MotionType.PASS_BOTTLE_LEFT_TO_RIGHT),
                        (ds.BodyParts.RightArm, ds.MotionType.PASS_BOTTLE_LEFT_TO_RIGHT)]
            self.partitioner, self.parts_IDs, self.trials, self.starts_ends = \
                    ds.load_recording(
                    ds.Recordings.exp4_pass_bottle_hold,
                    bodypart_motiontypes=bodypart_motiontypes,
                    max_chunks=5)
        elif self.param["dataset"] == "return-bottle-hold":
            if "parts" not in self.param or self.param["parts"] == 1:
                bodypart_motiontypes=[(ds.BodyParts.FullBody, ds.MotionType.PASS_BOTTLE_RIGHT_TO_LEFT)]
            elif self.param["parts"] == 3:
                bodypart_motiontypes=[
                        (ds.BodyParts.LeftArm, ds.MotionType.PASS_BOTTLE_RIGHT_TO_LEFT),
                        (ds.BodyParts.BodyNoArms, ds.MotionType.PASS_BOTTLE_RIGHT_TO_LEFT),
                        (ds.BodyParts.RightArm, ds.MotionType.PASS_BOTTLE_RIGHT_TO_LEFT)]
            self.partitioner, self.parts_IDs, self.trials, self.starts_ends = \
                    ds.load_recording(
                    ds.Recordings.exp4_pass_bottle_hold,
                    bodypart_motiontypes=bodypart_motiontypes,
                    max_chunks=5)
        
        self.trial = self.trials[0]
        self.trial.hold(self.param["hold"])
        self.nparts = np.max(self.parts_IDs) + 1
