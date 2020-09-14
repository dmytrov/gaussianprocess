import os
import time
import numpy as np
import dmp.discretedmp as ddmp
from validation.common import *
import validation.crossvalidate.common as vcc



class DMPCrossval(vcc.ModelLerner):

    def __init__(self, dirname, param, args):
        super(DMPCrossval, self).__init__(dirname, param, args)

        self.load_data()
        self.generate_BVHes()


    def train_model(self, training, validation):
        dmp = ddmp.DiscreteDMP(npsi=self.param["npsi"])
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


    def generate_BVHes(self):
        training = self.trial.training_data()
        validation = self.trial.heldout_data()

        t0 = time.time()
        dmp, predicted, errors = self.train_model(training, validation)
        t1 = time.time()

        errors = compute_errors(observed=validation, predicted=predicted)
        errors["timing"] = t1 - t0
        errors["settings"] = self.param

        # Make a BVH
        if self.partitioner is not None:
            self.partitioner.set_all_parts_data(predicted)
            self.partitioner.motiondata.write_BVH(
                    os.path.join(self.dirname, "final.bvh"))

        # Write the errors
        efn = os.path.join(self.dirname, "errors.pkl")
        with open(efn, "wb") as filehandle:
            pickle.dump(errors, filehandle)



