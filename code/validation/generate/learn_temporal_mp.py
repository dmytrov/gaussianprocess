import os
import sys
import time
import numpy as np
import temporalPrimitives.interpolateAndConcatenateMP as icmp
from validation.common import *
import validation.generate.common as vgc



class TMPLearner(vgc.ModelLerner):

    def __init__(self, dirname, param, args):
        super(TMPLearner, self).__init__(dirname, param, args)

        self.load_data()
        self.generate_BVHes()



    def train_TMP(self, training, validation):

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
        
        MPfile = "{}/MovementPrimitives_learned.pkl".format(self.dirname)
        try:
            print "trying to load primitive model..."
            df = open(MPfile, "rb")
            primModel = pickle.load(df)
            df.close()
        except:
            numprim = self.param["numprim"]
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


    def generate_BVHes(self):
        self.training_y = [t.copy() for t in self.trial.training_data()]
        for i in range(len(self.training_y)):
            validation = self.training_y[i]

            t0 = time.time()
            errors, predicted = self.train_TMP(training=self.training_y, 
                    validation=validation)
            t1 = time.time()
            errors["timing"] = t1-t0
            errors["settings"] = self.param
            
            # Make a BVH
            if self.partitioner is not None:
                self.partitioner.set_all_parts_data(predicted)
                self.partitioner.motiondata.write_BVH(
                        os.path.join(self.dirname, "final({}).bvh".format(i)))

            # Write the errors
            efn = os.path.join(self.dirname, "errors({}).pkl".format(i))
            with open(efn, "wb") as filehandle:
                pickle.dump(errors, filehandle)

        efn = os.path.join(self.dirname, "errors.pkl")
        with open(efn, "wb") as filehandle:
            pickle.dump(errors, filehandle)


