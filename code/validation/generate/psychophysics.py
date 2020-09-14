import os
if "DISPLAY" not in os.environ:
    import matplotlib
    print("No DISPLAY found. Switching to noninteractive matplotlib backend...")
    print("Old backend is: {}".format(matplotlib.get_backend()))
    matplotlib.use('Agg')
    print("New backend is: {}".format(matplotlib.get_backend()))
import utils.parallel.runner as pr
from validation.generate.learn_vcgpdm import VCGPDMLearner
from validation.generate.learn_dynamic_mp import DMPLearner
from validation.generate.learn_temporal_mp import TMPLearner
import validation.generate.common as vgc


class PsychophysicsGenerator(pr.JobRunner):
    
    def __init__(self):
        super(PsychophysicsGenerator, self).__init__(
                realpath=os.path.realpath(__file__),
                infostring="Generates all BVH files")
        

    def create_params(self):
        return vgc.create_params(full=True)


    def run_task(self, param):
        print("Task parameters: {}".format(param))
        dirname = os.path.join(self.args.dir, self.taskparams.to_dir_name(param))
        handler = {
                "vcgpdm": VCGPDMLearner, 
                "dmp": DMPLearner,
                "tmp": TMPLearner,
                }
        handler[param["model"]](dirname, param, self.args)
        
            

if __name__ == "__main__":
    PsychophysicsGenerator()
