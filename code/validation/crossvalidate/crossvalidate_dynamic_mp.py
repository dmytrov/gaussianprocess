import os
import pickle
import numpy as np
import matplotlib
if "DISPLAY" not in os.environ:
    print("No DISPLAY found. Switching to noninteractive matplotlib backend...")
    print("Old backend is: {}".format(matplotlib.get_backend()))
    matplotlib.use('Agg')
    print("New backend is: {}".format(matplotlib.get_backend()))
import matplotlib.pyplot as plt
import dmp.discretedmp as ddmp
from validation.common import *


def train_DMP(training, validation, settings):
    dmp = ddmp.DiscreteDMP(npsi=settings["npsi"])
    dmp.learn(training)
    dmp.reset()

    dmp.run(validation.shape[0])
    predicted = dmp.y_path.copy()
    predicted[np.isnan(predicted)] = 0.0
    errors = compute_errors(observed=validation, predicted=predicted)
    #print(predicted)
    #print(np.sum((validation-predicted)**2)
    #plt.plot(validation, color="blue")
    #plt.plot(predicted, color="orange")
    #plt.plot(validation-predicted, color="red")
    #plt.show()
    return dmp, predicted, errors


def run_DMP_crossvalidation(settings, trial, bvhpartitioner=None):
    errorsfilename = settings["directory"] + "/errors.pkl"
    if not os.path.exists(errorsfilename):
        if not os.path.exists(settings["directory"]):
            os.makedirs(settings["directory"])

        trial.hold(settings["hold"], settings["validation_seed_size"])  # hold-one crossvalidation
        training = trial.training_data()
        validation = trial.heldout_data()
        
        t0 = time.time()
        dmp, predicted, errors = train_DMP(training, validation, settings)
        t1 = time.time()

        errors = compute_errors(observed=validation,
                                predicted=predicted)
        errors["timing"] = t1 - t0
        errors["settings"] = settings

        # Make a BVH
        if bvhpartitioner is not None:
            bvhpartitioner.set_all_parts_data(predicted)
            bvhpartitioner.motiondata.write_BVH(settings["directory"] + "/final.bvh")

        # Write the errors
        with open(errorsfilename, "wb") as filehandle:
            pickle.dump(errors, filehandle)

        #plt.plot(validation)
        #plt.plot(predicted)
        #plt.show()


def create_model_iterator(id=1):
    miter = None
    if id == 0:
        miter = ModelIterator()
        miter.load_recording(ds.Recordings.sines, None)
    elif id == 1:
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
            ds.Recordings.exp3_walk,
            bodypart_motiontypes=[(ds.BodyParts.FullBody, ds.MotionType.WALK_PHASE_ALIGNED)],
            max_chunks=5)
    elif id == 4:
        miter = ModelIterator()
        miter.load_recording(
            ds.Recordings.exp4_pass_bottle_put,
            bodypart_motiontypes=[(ds.BodyParts.FullBody, ds.MotionType.PASS_BOTTLE_LEFT_TO_RIGHT)],
            max_chunks=5)
    
    miter.settings = {"recording_info": miter.recording.info,
                      "recording_filename": miter.recording.filename,
                      "validation_seed_size": 0,  # number of frames to be included into the training set
                      "directory": None,
                      "npsi": 50,  # DMP parameter
                      "hold": None,}
    miter.params_range = [("npsi", range(2, 20) + range(20, 51, 5)),
                          ("hold", range(miter.trial.nchunks()))]
    miter.directory = "../../../log/validation/crossvalidated/dmp/dataset({})".format(id)    
    return miter


if __name__ == "__main__":
    for dataset_id in (1, 2, 3, 4,):
        miter = create_model_iterator(dataset_id)
        miter.iterate_all_settings(run_DMP_crossvalidation)
