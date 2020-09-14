import os
import time
import numpy as np
import dmp.discretedmp as ddmp
from validation.common import *
import validation.generate.common as vgc



class DMPLearner(vgc.ModelLerner):

    def __init__(self, dirname, param, args):
        super(DMPLearner, self).__init__(dirname, param, args)

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
        self.training_y = [t.copy() for t in self.trial.training_data()]
        for i in range(len(self.training_y)):
            validation = self.training_y[i]
            t0 = time.time()
            dmp, predicted, errors = self.train_model(self.training_y, validation)
            t1 = time.time()

            errors = compute_errors(observed=validation,
                                    predicted=predicted)
            errors["timing"] = t1 - t0
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



