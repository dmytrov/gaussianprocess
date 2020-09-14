import os
if "DISPLAY" not in os.environ:
    import matplotlib
    print("No DISPLAY found. Switching to noninteractive matplotlib backend...")
    print("Old backend is: {}".format(matplotlib.get_backend()))
    matplotlib.use('Agg')
    print("New backend is: {}".format(matplotlib.get_backend()))
import utils.parallel.runner as pr
from validation.crossvalidate.crossval_vcgpdm import VCGPDMCrossval
from validation.crossvalidate.crossval_dynamic_mp import DMPCrossval
from validation.crossvalidate.crossval_temporal_mp import TMPCrossval
import validation.crossvalidate.common as vcc


class CrossvalidationRunner(pr.JobRunner):
    
    def __init__(self):
        super(CrossvalidationRunner, self).__init__(
                realpath=os.path.realpath(__file__),
                infostring="Runs crossvalidation model training")
        

    def create_params(self):
        return vcc.create_params(full=False)


    def run_task(self, param):
        print("Task parameters: {}".format(param))
        dirname = os.path.join(self.args.dir, self.taskparams.to_dir_name(param))
        handler = {
                "vcgpdm": VCGPDMCrossval, 
                "dmp": DMPCrossval,
                "tmp": TMPCrossval,
                }
        handler[param["model"]](dirname, param, self.args)
        
            

if __name__ == "__main__":
    CrossvalidationRunner()
