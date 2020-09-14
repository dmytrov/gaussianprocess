import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils.parallel.runner as pr
import validation.generate.common as vgc
import validation.common as vc



class StatsReader(pr.JobRunner):
    
    def __init__(self):
        super(StatsReader, self).__init__(
                realpath=os.path.realpath(__file__),
                infostring="Collects statistics",
                createpathes=False)
        

    def create_params(self):
        return vgc.create_params(full=True)


    def is_must_run(self, taskparam):
        return True
 

    def on_run_task_complete(self, i):
        pass

    
    def pre_run(self):
        self.nparts = 2
        self.elbos = {}
        self.stats = vc.ErrorStatsReader()
        

    def run_task(self, param):
        dirname = os.path.join(self.args.dir, 
                self.taskparams.to_dir_name(param))
        try:
            self.stats.read_learned_errors_from_file(
                    os.path.join(dirname, "errors.pkl"), param=param)
        except:
            pass

        try:
            filename = dirname + "/iter_3_alpha.txt"
            #print("Reading " + filename)
            with open(filename, "r") as filehandle:
                content = filehandle.readlines()
            full_elbo = float(content[1].strip())
            dyn_elbo = float(content[3].strip())
            lvm_elbo = float(content[5].strip())
            self.elbos[tuple(sorted(param.items()))] = {
                    "full_elbo": full_elbo, 
                    "dyn_elbo": dyn_elbo, 
                    "lvm_elbo": lvm_elbo, 
                    }
        except:
            pass
            

    def post_run(self):
        #print(self.stats.errs)
        print(self.args.dir)
        with open(os.path.join(self.args.dir, "combined_errors.pkl"), "wb") as filehandle:  
            pickle.dump(self.stats.errs, filehandle)

        with open(os.path.join(self.args.dir, "combined_elbo_stats.pkl"), "wb") as filehandle:  
            pickle.dump(self.elbos, filehandle)

        #n = len(self.elbos)
        #fig, axes = plt.subplots(nrows=1, ncols=n, \
        #        figsize=(5, 5), sharey=False)
        #for axis, elbo, i in zip(axes, self.elbos, range(n)):
        #    axis.imshow(elbo.T)
        #    axis.set_xlabel("dyn")
        #    axis.set_ylabel("lvm")
        #    axis.set_title("Parts: {}".format(i+1))
        #plt.savefig("ELBO.pdf")
        #plt.show()




if __name__ == "__main__":
    StatsReader()
