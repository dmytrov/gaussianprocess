import os
import sys
import time
import logging
import pickle
import numpy as np
import matplotlib
if "DISPLAY" not in os.environ:
    print("No DISPLAY found. Switching to noninteractive matplotlib backend...")
    print("Old backend is: {}".format(matplotlib.get_backend()))
    matplotlib.use('Agg')
    print("New backend is: {}".format(matplotlib.get_backend()))
import matplotlib.pyplot as plt
import dataset.mocap as ds
import temporalPrimitives.interpolateAndConcatenateMP as icmp
import numerical.numpyext.logger as npl
from validation.common import *


def train_TMP(training, validation, settings):

    def statusCallback(prefix, msg, vb, recon):
        """Callback function for monitoring learning.
        prefix: print at beginning of line
        msg: message to print
        vb: variational bound
        recon: reconstruction error"""
        print prefix, msg, vb, recon
        sys.stdout.flush()

    dt = 1.0 / 24
    timeDiscr = 100

    trainingData = []
    for td in training:
        step = dict()
        step["parameters"] = np.array([1.0, 1.0])
        step["timesteps"] = dt * np.arange(td.shape[0])
        step["trajectories"] = td.T
        trainingData += [step]
    
    MPfile = "{}/MovementPrimitives_learned.pkl".format(settings["directory"])
    try:
        print "trying to load primitive model..."
        df = open(MPfile, "rb")
        primModel = pickle.load(df)
        df.close()
    except:
        numprim = settings["numprim"]
        primModel = icmp.morphablePrimitiveWithInitialConditions(numprim, timeDiscr, dt)
        primModel.learnMovementPrimitives(
            trainingData, lambda msg, vb, recon: statusCallback("learning DOF:", msg, vb, recon))
        df = open(MPfile, "wb")
        pickle.dump(primModel, df)
        df.close()

    md = dict()
    md["parameters"] = np.array([1.0, 1.0])
    genDT = float(primModel.predictTimeWarpFactor(md["parameters"]))
    realTimeSteps = genDT * primModel.timeSteps
    generatingDT = genDT * (primModel.timeSteps[-1] - primModel.timeSteps[0]) / validation.shape[0]
    gentrajectory, gentimesteps = primModel.generateMovement(
                md, generatingDT, lambda msg, vb, recon: statusCallback("generating:", msg, vb, recon))

    observed = validation
    predicted = gentrajectory.T
    minsteps = min(observed.shape[0], predicted.shape[0])
    observed = observed[:minsteps, :]
    predicted = predicted[:minsteps, :]
    errors = compute_errors(observed=observed, predicted=predicted)
    errors["ELBO"] = float(primModel.bestVB)
    #plt.plot(observed, color="blue")
    #plt.plot(predicted, color="orange")
    #plt.show()
    return errors, predicted


def run_TMP_crossvalidation(settings, trial, bvhpartitioner=None):
    errorsfilename = settings["directory"] + "/errors.pkl"
    if not os.path.exists(errorsfilename):
        if not os.path.exists(settings["directory"]):
            os.makedirs(settings["directory"])

        trial.hold(settings["hold"], settings["validation_seed_size"])  # hold-one crossvalidation
        training = trial.training_data()
        validation = trial.heldout_data()

        t0 = time.time()
        errors, predicted = train_TMP(training=training, validation=validation, settings=settings)
        t1 = time.time()
        errors["timing"] = t1-t0
        errors["settings"] = settings

        # Make a BVH
        if bvhpartitioner is not None:
            bvhpartitioner.set_all_parts_data(predicted)
            bvhpartitioner.motiondata.write_BVH(settings["directory"] + "/final.bvh")

        with open(errorsfilename, "wb") as filehandle:
            pickle.dump(errors, filehandle)


def create_model_iterator(id=1):
    miter = None
    if id == 1:
        miter = ModelIterator()
        miter.load_recording(
            ds.Recordings.exp2_walk1,
            bodypart_motiontypes=[(ds.BodyParts.FullBody, ds.MotionType.WALK_PHASE_ALIGNED)])
    elif id == 2:
        miter = ModelIterator()
        miter.load_recording(
            ds.Recordings.exp2_walk_wave1,
            bodypart_motiontypes=[(ds.BodyParts.FullBody, ds.MotionType.WALK_WAVE_PHASE_ALIGNED)])
    elif id == 3:
        miter = ModelIterator()
        miter.load_recording(
            ds.Recordings.exp4_pass_bottle_put,
            bodypart_motiontypes=[(ds.BodyParts.FullBody, ds.MotionType.PASS_BOTTLE_LEFT_TO_RIGHT)],
            max_chunks=5)
    miter.settings = {"recording_info": miter.recording.info,
                      "recording_filename": miter.recording.filename,
                      "directory": None,
                      "validation_seed_size": 0,  # number of frames to be included into the training set
                      "hold": None,}
    miter.params_range = [("numprim", range(2, 11)), # + [13, 16, 19]),
                          ("hold", range(miter.trial.nchunks()))]
    miter.directory = "../../../log/validation/crossvalidated/tmp/dataset({})".format(id)
    return miter



if __name__ == "__main__":
    for dataset_id in (1, 2, 3):
        miter = create_model_iterator(id=dataset_id)
        miter.iterate_all_settings(run_TMP_crossvalidation)

