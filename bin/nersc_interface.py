from __future__ import absolute_import, print_function
from pspy import so_config, so_misc
from string import Template
import os 
from subprocess import Popen, PIPE

# parse command line input
so_config.argparser.add_argument('-i', '--iter',
    default='',
    help='iteration number')

so_config.argparser.add_argument('-v', '--verbose',
    default='f',
    help='enable verbose output')

so_config.argparser.add_argument('-t', '--task',
    required=True,
    help='task to run as defined in conf/nersc.ini')

so_config.argparser.add_argument('-d', '--dryrun',
    default='f',
    help='generate a job submission script without submitting it')

so_config.argparser.add_argument('-j', '--jobid',
    default=None,
    help="previous jobid")

so_config.argparser.add_argument('-dp', '--dependency',
    default='ok',
    help='dependency on previous job')


#so_config.argparser.print_help()
args = so_config.argparser.parse_args()

# temporary output file
output_temp_file = 'sbatch_script.sb'

# MPI RUN prefix
command_prefix_temp = "srun -n {} -c {} --cpu_bind=cores python" 

# CONSTANT
num_phy_core_per_node = 32

# Parse inputs from the command line
iterNum           = args.iter
is_dryrun         = so_misc.str2bool(args.dryrun)
is_verbose        = so_misc.str2bool(args.verbose) or is_dryrun
task_name         = args.task
prev_jobid        = args.jobid

# load config file
__confparser = so_config.configparser
so_config.load_config("nersc.ini")

# Load the default setting
__default_settings = dict(__confparser.items("DEFAULT_SLURM"))
settings = __default_settings.copy()

# Load inital tasks
__tasks   = dict(__confparser.items(task_name))

# main body of this script
def __task_helper(prev_jobid,__tasks):
    try:
        is_collection=so_misc.str2bool(__tasks.get('COLLECTION'))
    except:
        raise ValueError("Job is ill defined")

    if not is_collection:
        settings = __default_settings.copy()
        settings.update(__tasks)
        
        is_serial      = so_misc.str2bool(settings.get("SERIAL"))
        commands       = settings.get("COMMANDS")
        num_cmd        = len(commands.splitlines())

        if is_serial or num_cmd == 1:
            # this block is where job is actually being submitted
            num_node       = int(settings.get("SLURM_N"))
            is_hyperthread = so_misc.str2bool(settings.get("HYPERTHREAD"))
            tot_num_tasks  = int(settings.get("SLURM_n"))
            
            num_eff_core = num_node * num_phy_core_per_node
            num_eff_core *= 2 if is_hyperthread else 1

            core_per_tasks = num_eff_core / tot_num_tasks
            core_wasted    = num_eff_core % tot_num_tasks

            if core_wasted != 0: print("WARNING: %d cores are not used"%core_wasted)
            
            settings.update({"OMP_NUM_THREADS": core_per_tasks})

            # process commands
            commands       = settings.get("COMMANDS")
            commands       = commands.replace("-i","-i="+iterNum)
            
            command_prefix = command_prefix_temp.format(tot_num_tasks, core_per_tasks)
            commands       = commands.replace("python", command_prefix)
                
            settings.update({"COMMANDS": commands})

            # make script template and submit the job
            template_file = open('../configs/sbatch_temp.txt')
            template = Template( template_file.read() )
           
            script_cont = template.substitute(settings)

            if is_verbose: print(script_cont)

            # write the temporary file
            
            script = open(output_temp_file, 'w')
            script.write(script_cont)
            script.close()

            sbatch_extra = ''
            if not so_misc.is_empty(prev_jobid):
                dependency_types={"ok": "--dependency=afterok:{}", 'any':"--dependency=afterany:{}"}
                assert(args.dependency in dependency_types.keys())
    
                dependency_temp = dependency_types[args.dependency]
                sbatch_extra = dependency_temp.format(prev_jobid)

            command = None
            if so_misc.is_empty(sbatch_extra):
                command = ["sbatch", output_temp_file]
            else:
                command = ["sbatch", sbatch_extra, output_temp_file]

            #if is_verbose: print(command)

            if not is_dryrun:
                process = Popen(command, stdout=PIPE)
                (output, err) = process.communicate()
                exit_code = process.wait()

                # get the jobid
                prev_jobid = [int(s) for s in output.split() if s.isdigit()][0]
            else:
                # generate random id for testing purpose 
                import numpy.random as rand
                prev_jobid = rand.randint(1000)

            # clean up
            os.remove(output_temp_file)
        else:
            # if serial flag is set to false, we loop through individual commands
            partial_ids = []
            cmd_list    = commands.splitlines()
            
            for cmd in cmd_list:
                sub_task = __tasks.copy()
                sub_task.update({"COMMANDS": cmd})

                partial_id  = __task_helper(prev_jobid, sub_task)
                partial_ids.append(partial_id)

            prev_jobid = ":".join(partial_ids)

    else:
        # if we are dealing with a collection, we loop through each task under the collection
        try:
            sub_tasks = __tasks.get('TASKS')
        except:
            raise ValueError("Collection is ill defined")

        for sub_task in sub_tasks.split(';'):
            #print("current task %s"%sub_task)
            sub_task = dict(__confparser.items(sub_task.strip()))
            prev_jobid =  __task_helper(prev_jobid, sub_task)

    return str(prev_jobid)

# trigger the job
__task_helper(prev_jobid, __tasks)

