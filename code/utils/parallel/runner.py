import sys
import os
import time
import subprocess
import argparse
import textwrap



class Params():
    
    def __init__(self):
        self.items = [{}]  # list of dicts with batch params
        self.keys_order = []  # order of the batch params
        self.param_range = {}  # dict of params ranges
        self.args = None  # script command line args
        

    def times(self, name, range, when=None):
        if name not in self.keys_order:
            self.keys_order.append(name)
        self.param_range[name] = range
        res = []
        for item in self.items:
            if when is None or when(item):
                for value in range:
                    i = item.copy()
                    i[name] = value
                    res.append(i)
            else:
                res.append(item)
        self.items = res

    def to_dir_name(self, item):
        return "-".join(["{}({})".format(key, item[key]) for key in self.keys_order if key in item])

    @classmethod        
    def test(self):
        params = Params()
        params.times("model", range=["vcgpdm", "cgpdm", "dmp", "tmp"])
        params.times("dataset", range=["walking", "waving"])
        params.times("parts", range=[7, 8, 9], \
                when=lambda x: x["model"] in {"vcgpdm", "cgpdm"})
        params.times("dyn", range=[2, 3, 4,], \
                when=lambda x: x["model"] == "vcgpdm")
        params.times("lvm", range=[5, 6], \
                when=lambda x: x["model"] == "vcgpdm")        
        
        assert len(params.items) == 46
        for i in params.items:
            print(params.to_dir_name(i))


class JobRunner(object):
    def __init__(self, realpath, infostring=None, createpathes=True):
        self.realpath = realpath
        self.infostring = infostring
        self.createpathes = createpathes
        if self.infostring is None:
            self.infostring = ""
        
        self.taskparams = self.create_params()
        parser = argparse.ArgumentParser(
            #prog=__file__,
            description=self.infostring,
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument("--v", action='version', version='%(prog)s 0.1')
        parser.add_argument("--i", type=int, default=0,
                help="Run only i-th task (1-based indexing for SGE compatibility). 0 to run all tasks")
        parser.add_argument("--dir", type=str, 
                default="./output/" + \
                sys.argv[0][:-3] + \
                time.strftime("-%Y-%m-%d-%H.%M.%S"), 
                help="output directory")
        parser.add_argument("--mode", type=str, default="sp", 
                help="execution mode. \
                \n'SGE' to submit to Sun Grid Engine, \
                \n'mp' for serial separate multiprocessing, \
                \n'sp' for serial single process")
        parser.add_argument("--force", action="store_true", 
                help="force re-running all tasks")
        parser.add_argument("--dry", action="store_true", 
                help="dry (fast) run, no computations")
        parser.add_argument("--printdirs", action="store_true", 
                help="only print output directories names and exit")
        self.args = parser.parse_args()

        if self.args.printdirs:
            for param in self.taskparams.items:
                print(self.taskparams.to_dir_name(param))
            exit()

        if self.args.i > 0:
            self._run_single(self.args.i-1)
        elif self.args.mode == "SGE":
            self._run_SGE()
        elif self.args.mode == "mp":
            self._run_mp()
        elif self.args.mode == "sp":
            self._run_sp()
        else:
            raise ValueError("Unsupported mode: {}".format(self.args.mode))



    def create_params(self):
        """ Override this
        """
        return Params()


    def pre_run(self):
        """ Override this
        """
        print("pre_run()")

    def run_task(self, param):
        """ Override this
        """
        print("run_task()", param)        


    def post_run(self):
        """ Override this
        """
        print("post_run()")


    def _run_SGE(self):
        """ Create and submit a SGE job
        """
        # Check if Sun Grid Engine is available
        #if "SGE_ROOT" not in os.environ:
        #    raise ValueError("SGE is not found")
        
        # Create a SGE launch script
        ntasks = len(self.taskparams.items)

        prefixstr = textwrap.dedent(
                """                #!/bin/bash
                # Marc2 launch script.

                #$ -S /bin/bash
                #$ -cwd
                #$ -e ${HOME}/sge-temp
                #$ -o ${HOME}/sge-temp

                # max run time
                #$ -l h_rt=10:00:00

                # 2G RAM per CPU
                #$ -l h_vmem=2G

                # SMP, 8 CPU slots per task. 16 is the recommended max.
                #$ -pe smp* 8"""
                +
                """
                # Task array, 1-N
                #$ -t 1-{}
                """.format(ntasks)
                +
                """
                # Email
                #$ -m aes
                #$ -M ${USER}@staff.uni-marburg.de
                """)
        
        nodeconfigstr = textwrap.dedent("""
                # Load proper modules
                . /etc/profile.d/modules.sh
                module unload gcc/6.3.0
                module load gcc/7.2.0
                module load lalibs/openblas/gcc-7.2.0/0.2.20
                module load tools/python-2.7

                source ~/venv/bin/activate
                export PYTHONPATH="${PYTHONPATH}:${HOME}/projects/vCGPDM/code/"
                export OPENBLAS_NUM_THREADS=8
                export OMP_NUM_THREADS=8    

                # Use local node file system for compilation
                export THEANO_FLAGS=base_compiledir=${HPC_LOCAL}/${RANDOM}

                if [[ -z "${DISPLAY}" ]]; then
                    echo DISPLAY is not set, using "Agg" rendering backend.
                    export MPLBACKEND="Agg"
                fi

                """)
        nodeconfigstr += "cd {}\n".format(os.path.dirname(self.realpath))

        cmd = "python {} --i ${{SGE_TASK_ID}} --dir {}".format(
                os.path.basename(self.realpath),
                self.args.dir)
        if self.args.force:
            cmd += " --force"
        if self.args.dry:
            cmd += " --dry"

        sgescript = prefixstr + nodeconfigstr + cmd
        # Submit the job
        print("="*60)
        print(sgescript)
        print("="*60)
        jobscriptfilename = "./{}.sh".format(sys.argv[0])
        with open(jobscriptfilename, "w") as f:
            f.write(sgescript)

        if "SGE_ROOT" in os.environ:
            subprocess.call("qsub {}".format(jobscriptfilename), shell=True)
        else:
            raise ValueError("SGE is not found")


    def is_must_run(self, taskparam):
        out_dir = os.path.join(self.args.dir, 
                self.taskparams.to_dir_name(taskparam))
        finishedfilename = os.path.join(out_dir, "finished.txt")
        return self.args.force or not os.path.exists(finishedfilename)


    def _get_out_dir_name(self, i):
        taskparam = self.taskparams.items[i]
        return os.path.join(self.args.dir, 
                self.taskparams.to_dir_name(taskparam))


    def _ensure_path_exists(self, path):
        if self.createpathes and not os.path.exists(path):
            os.makedirs(path)


    def _run_single(self, i):
        taskparam = self.taskparams.items[i]
        if self.is_must_run(taskparam):
            self._ensure_path_exists(self._get_out_dir_name(i))
            self.run_task(taskparam)
            self.on_run_task_complete(i)
            

    def on_run_task_complete(self, i):
        taskparam = self.taskparams.items[i]
        out_dir = os.path.join(self.args.dir, 
                self.taskparams.to_dir_name(taskparam))

        finishedfilename = os.path.join(out_dir, "finished.txt")
        with open(finishedfilename, "w") as f:
            f.write("Finished.")


    def _run_mp(self):        
        """ Run multiple processes serially.
        """
        for i in range(len(self.taskparams.items)):
            cmd = "python {} --dir {} --i {}".format(
                    sys.argv[0],
                    self.args.dir, 
                    i+1)
            if self.args.force:
                cmd += " --force"
            if self.args.dry:
                cmd += " --dry"
            subprocess.call(cmd, shell=True)


    def _run_sp(self):
        """ Process all tasks in this process.
        """
        self.pre_run()
        for i in range(len(self.taskparams.items)):
            self._run_single(i)
        #for param in self.taskparams.items:
        #    self.run(param)
        self.post_run()



class TestJobRunner(JobRunner):
    def __init__(self, realpath, infostring=None):
        super(TestJobRunner, self).__init__(realpath, infostring)


    def create_params(self):
        params = Params()
        params.times("model", range=["vcgpdm", "cgpdm", "dmp", "tmp"])
        params.times("dataset", range=["walking", "waving"])
        params.times("parts", range=[7, 8, 9], \
                when=lambda x: x["model"] in {"vcgpdm", "cgpdm"})
        #params.times("dyn", range=[2, 3, 4,], \
        #        when=lambda x: x["model"] == "vcgpdm")
        #params.times("lvm", range=[5, 6], \
        #        when=lambda x: x["model"] == "vcgpdm")     
        return params

    def run_task(self, param):
        print("Running task with param ", param)


if __name__ == "__main__":   
    jr = TestJobRunner(realpath=os.path.realpath(__file__))
    
