import os
import ntpath
from time import strftime
import numpy as np
import bvhrwroutines.bvhrwroutines as br


dir_data = os.path.dirname(__file__) + "/../../../mocapdata/phasespace"


class MotionType(object):
    WALK = 0
    WALK_WAVE = 1
    WAVE = 2
    WALK_PHASE_ALIGNED = 3
    WALK_WAVE_PHASE_ALIGNED = 4
    SHAKE_RIGHTARM = 5
    SHAKE_ARMS = 6
    WALK_SHAKE_RIGHTARM = 7
    WALK_SHAKE_ARMS = 8
    PASS_BOTTLE_LEFT_TO_RIGHT = 9
    PASS_BOTTLE_RIGHT_TO_LEFT = 10

    TOY_SINES = 100


class BodyPart(object):
    def __init__(self, name, joints=None):
        self.name = name
        self.joints = []
        self.extend(joints)

    def extend(self, joints):
        if joints is not None:
            for joint in joints:
                if joint not in self.joints:
                    self.joints.append(joint)

    def include(self, bodyparts):
        for bodypart in bodyparts:
            self.extend(bodypart.joints)
        return self

    def exclude(self, bodyparts):
        to_exclude = []
        for bp in bodyparts:
            to_exclude.extend(bp.joints)
        self.joints = [joint for joint in self.joints if joint not in to_exclude]
        return self


class BodyParts(object):
    Pelvis = BodyPart(name="pelvis",
                      joints=["pelvis"])
    LeftLeg = BodyPart(name="left_leg",
                       joints=["pelvis_left_femur",
                               "left_femur_tibia",
                               "left_tibia_foot"])
    RightLeg = BodyPart(name="right_leg",
                        joints=["pelvis_right_femur",
                                "right_femur_tibia",
                                "right_tibia_foot"])
    Spine = BodyPart(name="spine",
                     joints=["pelvis_spine1",
                             "spine1"])
    Head = BodyPart(name="head",
                    joints=["spine2",
                            "neck"])
    LeftArm = BodyPart(name="left_arm",
                       joints=["left_manubrium",
                               "left_clavicle",
                               "left_humerus",
                               "left_radius"])
    RightArm = BodyPart(name="right_arm",
                        joints=["right_manubrium",
                                "right_clavicle",
                                "right_humerus",
                                "right_radius"])
    BodyNoArms = BodyPart(name="body_no_arms")
    Upper = BodyPart(name="upper")
    Lower = BodyPart(name="lower")
    FullBody = BodyPart(name="full_body")

BodyParts.Upper.include([BodyParts.Spine,
                         BodyParts.Head,
                         BodyParts.LeftArm,
                         BodyParts.RightArm])
BodyParts.Lower.include([BodyParts.Pelvis,
                         BodyParts.LeftLeg,
                         BodyParts.RightLeg])
BodyParts.FullBody.include([BodyParts.Lower,
                            BodyParts.Upper])
BodyParts.BodyNoArms.include([BodyParts.FullBody]) \
        .exclude([BodyParts.LeftArm,
                BodyParts.RightArm])


class MotionTrial(object):
    def __init__(self, data, starts_ends, motion_type=None):
        self.data = data
        self.starts_ends = starts_ends
        self.motion_type = motion_type
        self.training_starts_ends = None
        self.heldout_start_end = None

    def nchunks(self):
        return len(self.starts_ends)

    def hold(self, chunknumber=None, heldout_seed_size=5):
        """ Hold i-th data chunk
            Use chunknumber < 0 or None for all data
        """
        self.training_starts_ends = [se for i, se in enumerate(self.starts_ends) if i != chunknumber]
        if chunknumber >= 0:
            self.heldout_start_end = self.starts_ends[chunknumber] 
            heldoutstart = self.heldout_start_end[0]
            if heldout_seed_size > 0:
                self.training_starts_ends.append([heldoutstart, heldoutstart+heldout_seed_size])

    def training_data(self):
        return [self.data[s:e, :] for s, e in self.training_starts_ends]

    def heldout_data(self):
        if self.heldout_start_end is not None:
            s, e = self.heldout_start_end
            return self.data[s:e, :]
        else:
            return None


class SegmentLabel(object):
    def __init__(self, bodypart, motiontype, starts_ends):
        self.bodypart = bodypart
        self.motiontype = motiontype
        self.starts_ends = np.array(starts_ends, dtype=int)


class Recording(object):
    def __init__(self, filename, info=None, labels=None):
        self.filename = filename
        self.info = info
        self.labels = labels
        if self.labels is None:
            self.labels = []


class ToyDataGenerator(Recording):
    def __init__(self):
        super(ToyDataGenerator, self).__init__(None)

    def generate_data(self):
        raise NotImplementedError()


def _load_sines(nparts=1, nchunks=3, tchunk=30):
    T = nchunks * tchunk
    t = np.linspace(0.0, 2 * np.pi, num=T)
    f1 = nchunks * 2.0  # cycles
    f2 = nchunks * 2.0  # cycles
    f3 = nchunks * 3.14  # cycles
    y1 = np.vstack((5.0 * np.sin(f1 * t + 0.0), 5.0 * np.sin(f1 * t + 1.5),
                    4.1 * np.sin(f1 * t + 0.4), 4.1 * np.sin(f1 * t + 1.8),
                    3.1 * np.sin(f1 * t + 0.4), 3.1 * np.sin(f1 * t + 1.8),
                    2.1 * np.sin(f1 * t + 0.4), 2.1 * np.sin(f1 * t + 1.8),)).T
    y2 = np.vstack((5.0 * np.sin(f2 * t + 0.0), 5.0 * np.sin(f2 * t + 1.5),
                    4.1 * np.sin(f2 * t + 0.4), 4.1 * np.sin(f2 * t + 1.8),
                    3.1 * np.sin(f2 * t + 0.4), 3.1 * np.sin(f2 * t + 1.8),
                    2.1 * np.sin(f2 * t + 0.4), 2.1 * np.sin(f2 * t + 1.8),)).T
    y3 = np.vstack((5.0 * np.sin(f3 * t + 0.0), 5.0 * np.sin(f3 * t + 1.5),
                    4.1 * np.sin(f3 * t + 0.4), 4.1 * np.sin(f3 * t + 1.8),
                    3.1 * np.sin(f3 * t + 0.4), 3.1 * np.sin(f3 * t + 1.8),
                    2.1 * np.sin(f3 * t + 0.4), 2.1 * np.sin(f3 * t + 1.8),)).T
    ys = [y1, y2, y3]
    y = np.hstack(ys[:nparts])

    np.random.seed(0)
    # Add noise and random displacement
    y += 0.5 * np.reshape(np.random.normal(size=y.size), y.shape) \
         + 10.0 * np.random.normal(size=y.shape[1])

    parts_IDs = ([0]*8 + [1]*8 + [2]*8)[:nparts*8]
    starts_ends = [[i * tchunk, (i+1) * tchunk] for i in range(nchunks)]
    trial = MotionTrial(y, starts_ends, MotionType.TOY_SINES)
    return None, parts_IDs, [trial]


def load_sines(nparts=1, nchunks=3, tchunk=30):
    T = 1 * tchunk
    t = np.linspace(0.0, 2 * np.pi, num=T)
    f1 = 1 * 2.0  # cycles
    f2 = 1 * 2.0  # cycles
    f3 = 1 * 3.14  # cycles
    y1 = np.vstack((5.0 * np.sin(f1 * t + 0.0), 5.0 * np.sin(f1 * t + 1.5),
                    4.1 * np.sin(f1 * t + 0.4), 4.1 * np.sin(f1 * t + 1.8),
                    3.1 * np.sin(f1 * t + 0.4), 3.1 * np.sin(f1 * t + 1.8),
                    2.1 * np.sin(f1 * t + 0.4), 2.1 * np.sin(f1 * t + 1.8),)).T
    y2 = np.vstack((5.0 * np.sin(f2 * t + 0.0), 5.0 * np.sin(f2 * t + 1.5),
                    4.1 * np.sin(f2 * t + 0.4), 4.1 * np.sin(f2 * t + 1.8),
                    3.1 * np.sin(f2 * t + 0.4), 3.1 * np.sin(f2 * t + 1.8),
                    2.1 * np.sin(f2 * t + 0.4), 2.1 * np.sin(f2 * t + 1.8),)).T
    y3 = np.vstack((5.0 * np.sin(f3 * t + 0.0), 5.0 * np.sin(f3 * t + 1.5),
                    4.1 * np.sin(f3 * t + 0.4), 4.1 * np.sin(f3 * t + 1.8),
                    3.1 * np.sin(f3 * t + 0.4), 3.1 * np.sin(f3 * t + 1.8),
                    2.1 * np.sin(f3 * t + 0.4), 2.1 * np.sin(f3 * t + 1.8),)).T
    ys = [y1, y2, y3]
    y = np.hstack(ys[:nparts])
    y = np.vstack([y for i in range(nchunks)])

    np.random.seed(0)
    # Add noise and random displacement
    y += 0.5 * np.reshape(np.random.normal(size=y.size), y.shape) \
         + 10.0 * np.random.normal(size=y.shape[1])

    parts_IDs = ([0]*8 + [1]*8 + [2]*8)[:nparts*8]
    starts_ends = [[i * tchunk, (i+1) * tchunk] for i in range(nchunks)]
    trial = MotionTrial(y, starts_ends, MotionType.TOY_SINES)
    return None, parts_IDs, [trial], starts_ends


class SineDataGenerator(ToyDataGenerator):
    def __init__(self, nparts=1):
        super(SineDataGenerator, self).__init__()
        self.nparts = nparts


    def generate_data(self):
        return load_sines(self.nparts)



recordings = {
    "sines": SineDataGenerator(nparts=3),
    "exp1_walk1": Recording(filename=dir_data + "/2016.02.26_bjoern/IC-001_skeleton.bvh",
                      info="Box experiment. Bjoern wakling freely",
                      labels=[SegmentLabel(BodyParts.FullBody,
                                          MotionType.WALK, 
                                          starts_ends=[[28,58],  # ccw
                                                       [234,265],  # cw
                                                       [456,486],  # ccw
                                                       [668,717],  # cw
                                                       [907,937],  # ccw
                                                       [1125,1155],]  # cw
                                          )
                             ]
                     ),
    "exp2_walk1": Recording(dir_data + "/2016.05.03_bjoern/003_skeleton.bvh",
                            info="Wave experiment. Bjoern wakling freely",
                            labels=[SegmentLabel(BodyParts.FullBody,
                                                 MotionType.WALK_PHASE_ALIGNED,
                                                 starts_ends=[[85, 124],  # clean
                                                              [233, 275],  # clean
                                                              [373, 413],  # clean
                                                              [502, 542],  # clean
                                                              [629, 668], ]  # clean
                                                 ),
                                    SegmentLabel(BodyParts.FullBody,
                                                 MotionType.WALK,
                                                 starts_ends=[[85, 120],  # clean
                                                              [225, 275],  # clean
                                                              [380, 410],  # clean
                                                              [490, 535],  # clean
                                                              [625, 665],]  # clean
                                                 )
                                    ]
                            ),
    "exp2_walk_wave1": Recording(filename=dir_data + "/2016.05.03_bjoern/011_skeleton.bvh",
                      info="Wave experiment. Bjoern walks while waving both his arms",
                      labels=[SegmentLabel(BodyParts.FullBody,
                                          MotionType.WALK_WAVE_PHASE_ALIGNED,
                                          starts_ends = [  # [left foot touch, right foot touch]
                                                        [1916, 1964],  # clean, slight drift
                                                        [3051, 3100],  # clean, slight drift
                                                        [1320, 1368],  # clean, slight drift, still the best 
                                                        [2675, 2724], ]  # clean, no drift
                                          ),
                              SegmentLabel(BodyParts.Lower,
                                          MotionType.WALK,
                                          starts_ends = [##[60, 95],    # lots of artefacts around
                                                        ##[264, 289],  # lots of artefacts around
                                                        ##[400, 440],  # lots of artefacts around
                                                        ##[600, 625],  # lots of artefacts around
                                                        ##[740, 774],  # lots of artefacts around
                                                        #[1135, 1200],  # clean, drift
                                                        #[1520, 1580],  # clean, drift
                                                        #[1750, 1790],  # clean, drift
                                                        [1920, 1965],  # clean, slight drift
                                                        [3035, 3100],  # clean, slight drift
                                                        [1350, 1380],  # clean, slight drift, still the best 
                                                        [2665, 2730], ]  # clean, no drift
                                          ),
                                SegmentLabel(BodyParts.Upper,
                                          MotionType.WAVE,
                                          starts_ends = [##[60, 95],    # lots of artefacts around
                                                        ##[264, 289],  # lots of artefacts around
                                                        ##[400, 440],  # lots of artefacts around
                                                        ##[600, 625],  # lots of artefacts around
                                                        ##[740, 774],  # lots of artefacts around
                                                        #[1135, 1200],  # clean, drift
                                                        #[1520, 1580],  # clean, drift
                                                        #[1750, 1790],  # clean, drift
                                                        [1920, 1965],  # clean, slight drift
                                                        [3035, 3100],  # clean, slight drift
                                                        [1350, 1380],  # clean, slight drift, still the best 
                                                        [2665, 2730], ]  # clean, no drift
                                          )
                             ]
                          ),
    "exp3_walk": Recording(filename=dir_data + "/2018.01.30_olaf/walking-01_skeleton.bvh",
                      info="Walking. Olaf walks in both directions. Only one direction is selected.",
                      labels=[SegmentLabel(BodyParts.FullBody,
                                          MotionType.WALK_PHASE_ALIGNED,
                                          starts_ends = [  # [right foot detouch, 8 steps]
                                                        ## [30, 138], # don't use the first one
                                                        [609, 713], 
                                                        [1144, 1244],
                                                        [1672, 1774], 
                                                        [2245, 2342],
                                                        [2783, 2882],
                                                        [3339, 3437],
                                                        [3859, 3959],
                                                        ## [4393, 4482], # too short
                                                        ## [4972, 5060], # too short
                                                        [5622, 5722],
                                                        [7126, 7226],
                                                        [7637, 7737],
                                                        [8165, 8270],
                                                        [8689, 8792],
                                                        [9336, 9436],
                                                        [9890, 9990],
                                                        [10448, 10549],
                                                        [11050, 11150],
                                                        [11654, 11764],
                                                        [12212, 12316],
                                                        [12784, 12889],
                                                        [13503, 13604],
                                                        [14090, 14194],
                                                        [14617, 14721],
                                                        [15130, 15237],
                                                        [15708, 15818],
                                                        [16233, 16344],
                                                        [16788, 16895],
                                                        [17368, 17471],
                                                        [17887, 17992],
                                                        [18447, 18550],
                                                        [19023, 19126],
                                                        [19689, 19798],
                                                        [20211, 20311],
                                                        [20739, 20843],
                                                        [21252, 21355],
                                                        [21783, 21883],
                                                        [22336, 22437],
                                                        [22873, 22977],
                                                        [23413, 23513],
                                                        [24354, 24455],
                                                        [24909, 25012],
                                                        [25444, 25544],
                                                        [25938, 26039],
                                                        [26468, 26569],
                                                        [26978, 27079],
                                                        [27475, 27577],
                                                        [27948, 28052],
                                                        [28434, 28537],
                                                        [28965, 29065],
                                                        [29465, 29568],
                                                        ] 
                                          )]),
    "exp3_protest_stand_1arm": Recording(filename=dir_data + "/2018.01.30_olaf/protesting_standing_1arm-01_skeleton.bvh",
                      info="Standing and protesting with right arm.",
                      labels=[SegmentLabel(BodyParts.Upper,
                                          MotionType.SHAKE_RIGHTARM,
                                          starts_ends = [  # [arm max back, arm max front after 10 cycles]
                                                        ## [400, 537], # don't use the first one
                                                        [609, 713], 
                                                        [629, 765],
                                                        [1418, 1551],
                                                        [1730, 1861],
                                                        [1952, 2088],
                                                        [2188, 2322],
                                                        [2402, 2536],
                                                        [2721, 2856],
                                                        [2938, 3072],
                                                        [3162, 3301],
                                                        [3643, 3781],
                                                        [3876, 4020],
                                                        [4120, 4264],
                                                        [4372, 4509],
                                                        [4604, 4742],
                                                        # ... changed pose, cleaner data
                                                        [7275, 7404],
                                                        [7491, 7630],
                                                        [7843, 8088],
                                                        [8473, 8603],
                                                        [8742, 8881],                                                        
                                                        ] 
                                          )]),
    "exp3_protest_stand_2arms": Recording(filename=dir_data + "/2018.01.30_olaf/protesting_standing_2arms-01_skeleton.bvh",
                      info="Standing and protesting with both arms.",
                      labels=[SegmentLabel(BodyParts.Upper,
                                          MotionType.SHAKE_ARMS,
                                          starts_ends = [  # [arm max back, arm max front after 10 cycles]
                                                        ## [162, 286], # don't use the first one
                                                        [394, 524],
                                                        [622, 756],
                                                        [866, 995],
                                                        [1170, 1296],
                                                        [1419, 1546],
                                                        [1673, 1798],
                                                        [1909, 2033],
                                                        [2166, 2290],
                                                        [2427, 2553],
                                                        [2692, 2816],
                                                        [2950, 3074],
                                                        ] 
                                          )]),                                                                                
    "exp3_protest_walk_1arm": Recording(filename=dir_data + "/2018.01.30_olaf/protesting_walking_1arm-01_skeleton.bvh",
                      info="Walking and protesting with right arm.",
                      labels=[SegmentLabel(BodyParts.FullBody,
                                          MotionType.WALK_SHAKE_RIGHTARM,
                                          starts_ends = [  # [left foot release, after 3 steps]
                                                        ## [208, 288], # don't use the first one
                                                        [697, 779],
                                                        [1518, 1598],
                                                        [1744, 1825],
                                                        [2004, 2082],
                                                        [2268, 2348],
                                                        [2492, 2571],
                                                        [2742, 2824],
                                                        [2987, 3067],
                                                        [3226, 3307],
                                                        [3468, 3550],
                                                        [3721, 3804],
                                                        [3993, 4073],
                                                        [4270, 4350],
                                                        [4527, 4610],
                                                        [4778, 4862],
                                                        [5044, 5123],
                                                        # ...
                                                        ] 
                                          )]),            
   "exp3_protest_walk_2arms": Recording(filename=dir_data + "/2018.01.30_olaf/protesting_walking_2arms-01_skeleton.bvh",
                      info="Walking and protesting with both arm.",
                      labels=[SegmentLabel(BodyParts.FullBody,
                                          MotionType.WALK_SHAKE_ARMS,
                                          starts_ends = [  # [left foot release, after 3 steps]
                                                        ## [103, 182], # don't use the first one
                                                        [367, 445],
                                                        [664, 743],
                                                        [920, 1001],
                                                        [1209, 1287],
                                                        [1477, 1556],
                                                        [1752, 1830],
                                                        [2028, 2108],
                                                        [2294, 2372],
                                                        [2532, 2610],
                                                        [2787, 2867],
                                                        [3024, 3103],
                                                        [3290, 3370],
                                                        [3534, 3614],
                                                        [3790, 3871],
                                                        [4061, 4142],
                                                        [4330, 4408],
                                                        # ...
                                                        ] 
                                          )]),  
    "exp4_pass_bottle_put": Recording(filename=dir_data + "/passtheobject/01/01-Teil1-A_skeleton.bvh",
                      info="Passing a bottle, put the bottle in between the passes",
                      labels=[SegmentLabel(BodyParts.FullBody,
                                          MotionType.PASS_BOTTLE_LEFT_TO_RIGHT,
                                          starts_ends = [  # [left arm starts moving, right arm stops]
                                                        [198, 356],
                                                        [511, 638],
                                                        [812, 955],
                                                        [1145, 1281],
                                                        [1439, 1573],
                                                        [1727, 1857],
                                                        [2012, 2144],
                                                        [2305, 2444],
                                                        [2578, 2708],
                                                        [2869, 2998],
                                                        # ...
                                                        ] 
                                          ),
                            SegmentLabel(BodyParts.FullBody,
                                          MotionType.PASS_BOTTLE_RIGHT_TO_LEFT,
                                          starts_ends = [  # [right arm starts moving, left arm stops]
                                                        [364, 490],
                                                        [652, 794],
                                                        [974, 1128],
                                                        [1291, 1425],
                                                        [1585, 1720],
                                                        [2155, 2296],
                                                        [2458, 2568],
                                                        [2718, 2861],
                                                        [3008, 3149],                                                      
                                                        # ...
                                                        ] 
                                          )]), 
    "exp4_pass_bottle_hold": Recording(filename=dir_data + "/passtheobject/01/01-Teil2-AB_skeleton.bvh",
                      info="Passing a bottle, hold the bottle in between the passes",
                      labels=[SegmentLabel(BodyParts.FullBody,
                                          MotionType.PASS_BOTTLE_LEFT_TO_RIGHT,
                                          starts_ends = [  # [left arm starts moving, right arm stops]
                                                        [518, 595],
                                                        [687, 761],
                                                        [847, 924],
                                                        [1017, 1099],
                                                        [1193, 1276],
                                                        [1373, 1460],
                                                        # ...
                                                        ] 
                                          ),
                            SegmentLabel(BodyParts.FullBody,
                                          MotionType.PASS_BOTTLE_RIGHT_TO_LEFT,
                                          starts_ends = [  # [right arm starts moving, left arm stops]
                                                        [393, 512],
                                                        [602, 679],
                                                        [764, 844],
                                                        [929, 1013],
                                                        [1104, 1190],
                                                        [1283, 1366],                                                      
                                                        # ...
                                                        ] 
                                          )]),                                                                                                                                                                                                        

}


class Recordings(object):
    sines = recordings["sines"]
    exp1_walk1 = recordings["exp1_walk1"]
    exp2_walk1 = recordings["exp2_walk1"]
    exp2_walk_wave1 = recordings["exp2_walk_wave1"]
    exp3_walk = recordings["exp3_walk"]
    exp3_protest_stand_1arm = recordings["exp3_protest_stand_1arm"]
    exp3_protest_walk_1arm = recordings["exp3_protest_walk_1arm"]
    exp3_protest_stand_2arms = recordings["exp3_protest_stand_2arms"]
    exp3_protest_walk_2arms = recordings["exp3_protest_walk_2arms"]
    exp4_pass_bottle_put = recordings["exp4_pass_bottle_put"]
    exp4_pass_bottle_hold = recordings["exp4_pass_bottle_hold"]


def make_common_starts_ends(labels, bodypart_motiontypes):
    """
    bodypart_motiontypes: [(bodypart, motiontype)] list of tuples
    returns: common starts_ends
    """
    assert isinstance(labels[0], SegmentLabel)
    sls = []
    for bpart, mtype in bodypart_motiontypes:
        ses = []
        for label in labels:
            if set(bpart.joints) <= set(label.bodypart.joints) and mtype == label.motiontype:
                ses.append(label.starts_ends)
        if len(ses) > 0:
            sls.append(np.vstack(ses))
        else:
            return np.array([])  # no labels found for the body part

    a = np.zeros(np.max([np.max(sl) for sl in sls])+1)
    for sl in sls:
        for se in sl:
            a[se[0]:se[1]] = 1
    da = a[1:] - a[:-1]
    starts = np.where(da > 0)[0] + 1
    ends = np.where(da < 0)[0] + 1
    starts_ends = np.vstack([starts, ends]).T
    return starts_ends



class DataSelectionMode(object):
    Intersection = 0  # select data where the full set of motion types is available


def load_recording(recording,
                   bodypart_motiontypes,
                   selection_mode=DataSelectionMode.Intersection,
                   max_chunks=None):
    """
    Load BVH data from the recording and select data subset.
    bodypart_motiontypes: [(bodypart, motiontype)] list of tuples
    """
    if isinstance(recording, ToyDataGenerator):
        return recording.generate_data()
    assert isinstance(recording, Recording)
    motiondata = br.MotionData()
    motiondata.read_BVH(recording.filename)
    partitioner = br.BVH_Partitioner(motiondata)
    for bodypart, motiontype in bodypart_motiontypes:
        partitioner.add_part(bodypart.name, bodypart.joints)
    #for label in recording.labels:
    #    partitioner.add_part(label.bodypart.name, label.bodypart.joints)
    y, parts_IDs = partitioner.get_all_parts_data_and_IDs()
    if selection_mode == DataSelectionMode.Intersection:
        starts_ends = make_common_starts_ends(
            recording.labels, bodypart_motiontypes)
        if max_chunks is not None:
            starts_ends = starts_ends[:max_chunks]
        trials = [MotionTrial(y, starts_ends)]
    return partitioner, parts_IDs, trials, starts_ends


def save_recording_chunks(filename, starts_ends, outdir=None):
    if outdir is None:
        outdir = "."
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    motiondata = br.MotionData()
    motiondata.read_BVH(filename)
    partitioner = br.BVH_Partitioner(motiondata)
    for bodypart in [BodyParts.FullBody]:
        partitioner.add_part(bodypart.name, bodypart.joints)
    y, parts_IDs = partitioner.get_all_parts_data_and_IDs()
    for start, end in starts_ends:
        partitioner.set_all_parts_data(y[start:end, :])
        outputfilename = "{}/{}-({}-{}).bvh".format(outdir, ntpath.split(filename)[1][:-4], start, end)
        print("Writting file {}".format(outputfilename))
        motiondata.write_BVH(outputfilename)


def save_recording_copy(recording, labelsID, outdir=None):
    assert recording.labels[labelsID].bodypart == BodyParts.FullBody
    save_recording_chunks(filename=recording.filename, 
        starts_ends=recording.labels[labelsID].starts_ends,
        outdir=outdir)



def select_sequences(partitioner, starts_ends):
    y, parts_IDs = partitioner.get_all_parts_data_and_IDs()
    sequences = [y[start:end] for start, end in starts_ends]
    return sequences

if __name__ == "__main__":
    recording = Recordings.exp3_walk
    save_recording_copy(recording)
    exit()
    save_recording_chunks(filename=recording.filename, 
        starts_ends=recording.labels[0].starts_ends,
        outdir="training")
    exit()
    
    partitioner, parts_IDs, trials, starts_ends = load_recording(
        recording=Recordings.exp1_walk1,
        #bodypart_motiontypes=[(BodyParts.FullBody, MotionType.WALK)])
        bodypart_motiontypes=[(BodyParts.Upper, MotionType.WALK),
                              (BodyParts.Lower, MotionType.WALK)])
    trial = trials[0]
    trial.hold(0)
    print("Full data ", trial.data.shape)
    for td in trial.training_data():
        print("Training ", td.shape)
    print("Held-out ", trial.heldout_data().shape)
    print("parts_IDs", parts_IDs)

    
