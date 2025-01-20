import argparse
import datetime
import importlib
import logging
import os
import shutil
import subprocess
import time
from copy import deepcopy
from itertools import product
from pathlib import Path

import yaml
from dateutil.parser import parse as time_parser
from pspy import so_dict
from simple_slurm import Slurm

pspipe_root = Path(__file__).parents[1]
placeholders = {
    "__pspipe_root__": str(pspipe_root),
} | {f"${key}": value for key, value in os.environ.items()}


def yaml_concat(loader, node):
    return os.path.join(*loader.construct_sequence(node))


# ast.literal_eval is too restrictive here. Use eval given the user knows what he is doing using
# !eval tag in the yaml file
def yaml_eval(loader, node):
    return eval(loader.construct_scalar(node))


def yaml_sub(loader, node):
    value = loader.construct_scalar(node)
    for place, holder in placeholders.items():
        value = value.replace(place, holder)
    return value


yaml.add_constructor("!concat", yaml_concat)
yaml.add_constructor("!eval", yaml_eval)
yaml.add_constructor("tag:yaml.org,2002:str", yaml_sub)


def main(args=None):

    # Homemade tools
    plural = lambda n: "s" if n > 1 else ""

    def parse_time(t, human=True):
        t = t if isinstance(t, str) else str(datetime.timedelta(seconds=round(t)))
        h, m, s = map(int, t.split(":"))
        time = f"{s} second" + plural(s) if human else f"{h:02d}:{m:02d}:{s:02d}"
        if human:
            for t, name in zip([m, h], ["minute", "hour"]):
                if not t:
                    continue
                time = f"{t} {name}" + plural(t) + ", " + time
        return time

    parser = argparse.ArgumentParser(description="A python pipe for CMB analysis")
    parser.add_argument(
        "-d", "--debug", help="Enable debug level", default=False, action="store_true"
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Overwrite previous output, if it exists",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-p", "--pipeline", help="Set pipeline file", required=True, type=os.path.realpath
    )
    parser.add_argument("-c", "--config", help="Set config/dict file", type=os.path.realpath)
    parser.add_argument(
        "--product-dir", help="Overload the product directory of yaml file", type=os.path.realpath
    )
    parser.add_argument(
        "-t",
        "--test",
        help="Only test the pipeline chain no run",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--var",
        help="Set variable to be overloaded in config file",
        action="append",
        default=list(),
    )
    parser.add_argument(
        "-b",
        "--batch",
        help="Execute pipeline in slurm batch mode",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-s", "--skip-dict-check", help="Skip global dict check", default=False, action="store_true"
    )

    args = parser.parse_args(args)

    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    logging.info("Reading pipeline file...")
    with open(args.pipeline, "r") as f:
        pipeline_dict = yaml.load(f, yaml.Loader)
    # Add previous args to argparse Namespace
    if args_from_yaml := pipeline_dict.get("args"):
        for k, v in args_from_yaml.items():
            setattr(args, k, v)
    pipeline_dict["args"] = vars(args)

    config_dict = so_dict.so_dict()
    if not (config_file := args.config or pipeline_dict.get("config_file")):
        if config_file is not False:
            logging.error(
                "Missing config/dict file ! "
                + "Either set 'config_file' in the pipeline file or in command line."
            )
            raise SystemExit()
    else:
        logging.info("Reading & updating configuration file...")
        config_dict.read_from_file(config_file)
        config_dict.update(dict(debug=pipeline_dict.get("debug", False) or args.debug))

    # Changing configuration value
    variables = pipeline_dict.get("variables", [])
    for var in variables:
        args.var += [f"{var}={variables[var]}"]
    logging.debug(f"args.var = {args.var}")
    for arg in set(args.var):
        keys, val = arg.split("=")
        logging.warning(f"Updating '{keys.replace('.', '/')}' value to {val}")
        do_replace = True
        d = config_dict
        for key in keys.split("."):
            if key not in d:
                logging.warning(f"Key '{key}' not found in config ! No replacement.")
                do_replace = False
                break
            if not isinstance(d[key], dict):
                break
            d = d[key]
        if do_replace:
            try:
                d[key] = eval(val)
            except:
                d[key] = val
    logging.debug(f"Configuration dict: {config_dict}")

    # Set production directory
    if not (product_dir := args.product_dir or pipeline_dict.get("product_dir")):
        logging.error(
            "Missing production directory ! "
            + "Either set 'product_dir' in the pipeline file or in command line."
        )
        raise SystemExit()
    product_dir = os.path.realpath(product_dir)
    if args.force:
        shutil.rmtree(product_dir, ignore_errors=True)
    os.makedirs(product_dir, exist_ok=True)
    os.chdir(product_dir)
    logging.info(f"Produced files are stored within {product_dir}")

    if config_file:
        updated_dict_file = os.path.join(
            product_dir, "{}_updated{}".format(*os.path.splitext(os.path.basename(config_file)))
        )
        if os.path.isfile(updated_dict_file):
            # Make sure the current and updated dict are the same
            updated_dict = so_dict.so_dict()
            updated_dict.read_from_file(updated_dict_file)
            exclude_patterns = ["debug"]
            get_dict = lambda d: {k: v for k, v in d.items() if k not in exclude_patterns}
            d1 = get_dict(config_dict)
            d2 = get_dict(updated_dict)
            if d1 != d2:
                diff_keys = {k for k in d1.keys() if d1.get(k) != d2.get(k)}
                msg = (
                    "Current setup is different from the previous one ! "
                    + f"The following keys '{diff_keys}' are different !"
                )
                if args.skip_dict_check:
                    logging.warning(msg)
                else:
                    logging.error(msg + " Make sure to run same configuration or use --force.")
                    raise SystemExit()
        config_dict.write_to_file(updated_dict_file)

    info = "Pipeline runs with the following software version:"
    modules = [
        "camb",
        "cobaya",
        "ducc0",
        "fgspectra",
        "mflike",
        "numpy",
        "pixell",
        "pspy",
        "pspipe_utils",
        "pspipe",
        "scipy",
    ]
    for m in modules:
        version = importlib.import_module(m).__version__
        pipeline_dict.setdefault("modules", {}).update({m: version})
        info += f"\n  - {m} {version}"
    logging.info(info)

    # Make sure every modules params are dict
    pipeline = pipeline_dict.setdefault("pipeline", {})
    pipeline |= {k: v if v else {} for k, v in pipeline.items()}

    # Check for matrix elements
    matrix_pipeline = dict()
    for module, params in pipeline.items():
        if matrix := params.pop("matrix", None):
            module_kwargs = params.pop("kwargs", "")
            if isinstance(module_kwargs, dict):
                module_kwargs = " ".join(f"{k} {v if v else ''}" for k, v in module_kwargs.items())
            runs_kwargs = {
                module.format(
                    **(fmt := {k: v for k, v in zip(matrix.keys(), items)})
                ): module_kwargs.format(**fmt)
                for items in product(*matrix.values())
            }
            matrix_pipeline.update({k: params | dict(kwargs=v) for k, v in runs_kwargs.items()})
        else:
            matrix_pipeline.update({module: params})

    # Get updated info
    updated_yaml_file = os.path.join(
        product_dir, "{}_updated{}".format(*os.path.splitext(os.path.basename(args.pipeline)))
    )
    if os.path.isfile(updated_yaml_file):
        with open(updated_yaml_file, "r") as f:
            updated_pipeline_dict = yaml.load(f, yaml.Loader)
        for module, params in matrix_pipeline.items():
            if params.get("force", False) or args.force:
                continue
            updated_params = updated_pipeline_dict.get("pipeline", {}).get(module, {})
            params |= {k: updated_params.get(k) for k in ["done", "duration"]}

    pipeline = deepcopy(matrix_pipeline)
    updated_pipeline_dict = deepcopy(pipeline_dict)
    updated_pipeline_dict.update(pipeline=pipeline)

    # Slurm default parameters
    slurm_nnodes = int(os.environ.get("SLURM_NNODES", 1))
    slurm_cpus_on_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 256))
    get_cpus_per_task = lambda ntasks: (
        slurm_cpus_on_node
        if ntasks < slurm_nnodes
        else (slurm_nnodes * slurm_cpus_on_node) // ntasks
    )
    default_kwargs = dict(ntasks=1, cpus_per_task=256)
    if args.batch:
        logging.info("Pipeline will be run in batch mode")
        slurm_kwargs = updated_pipeline_dict.get("slurm", {})
        precmd = slurm_kwargs.pop("precmd", "")
        postcmd = slurm_kwargs.pop("postcmd", "")
        slurm = Slurm(**slurm_kwargs)
        slurm.add_cmd(precmd)
        # Prepare log directory
        if output := slurm_kwargs.get("output"):
            os.makedirs(os.path.dirname(output), exist_ok=True)

    for module, params in pipeline.items():
        # Get job status
        if done := params.get("done", False):
            logging.info(
                f"Module '{module}' already processed {done} "
                + f"(running time: {parse_time(params.get('duration'))}), skipping it."
            )
            continue

        if not (cmd := params.get("cmd")):
            # Make sure the script exists
            script_base_dir = pipeline_dict.get("script_base_dir") or os.path.join(
                pspipe_root, "project/ACT_DR6/python"
            )
            script_file, ext = os.path.splitext(params.get("script_file", module))
            script_file = os.path.join(script_base_dir, script_file) + (ext or ".py")
            if not os.path.exists(script_file):
                raise ValueError(f"File {script_file} does not exist!")

        # Prepare slurm job if any
        slurm_params = params.get("slurm") or dict()
        slurm_kwargs = deepcopy(default_kwargs)
        slurm_kwargs.update(cpus_per_task=get_cpus_per_task(slurm_params.get("ntasks", 1)))
        slurm_kwargs.update(slurm_params)

        if not args.batch:
            # Create slurm instance
            need_slurm = params.get("slurm", False)
            slurm = Slurm(**slurm_kwargs) if need_slurm else None
            # Check for slurm need
            if need_slurm and not os.environ.get("SLURM_NNODES"):
                logging.error("Pipeline needs to be run on slurm ! Log first with salloc.")
                raise SystemExit()

            # Make sure we have enough nodes
            slurm_nnodes = int(os.environ.get("SLURM_NNODES", 1))
            if (nodes := slurm_params.get("nodes")) and nodes > slurm_nnodes:
                logging.error(
                    f"Module '{module}' can not be run on only {slurm_nnodes} nodes "
                    + f"(needs {nodes} nodes)"
                )
                raise SystemExit()

            # Make sure we have enough time
            if minimal_time := params.get("minimal_needed_time"):
                squeue_cmd = subprocess.run(
                    ["squeue", "-h", "-j", os.environ.get("SLURM_JOBID"), "-o", '"%L"'],
                    capture_output=True,
                    text=True,
                )
                remaining_time = squeue_cmd.stdout.replace("\n", "").replace('"', "")
                if len(remaining_time.split(":")) < 3:
                    remaining_time = "00:" + remaining_time
                if time_parser(remaining_time) < time_parser(minimal_time):
                    logging.error(f"Module {module} does not have enough time in the current node")
                    raise SystemExit()

        logging.info(
            ("Preparing" if args.batch else "Running")
            + f" '{module}' script on {(ntasks := slurm_kwargs.get('ntasks'))} task"
            + plural(ntasks)
            + f" ({(cpus := slurm_kwargs.get('cpus_per_task'))} cpu"
            + plural(cpus)
            + " per task)"
        )

        # Get additionnal parameters to pass to script
        if kwargs := params.get("kwargs", ""):
            logging.info(f"Passing the following arguments to the module: {kwargs}")

        if args.test:
            continue

        if not cmd:
            cmd = script_file
            # If no extension is provided, assume it's a python file
            if not ext:
                cmd = "python -u " + cmd
        # Append kwargs to command line
        cmd = f"{cmd} {kwargs} {updated_dict_file if config_file else ''}"
        logging.debug(f"Command line: {cmd}")

        srun_cmd = f"OMP_NUM_THREADS={cpus} srun --cpu-bind=cores"
        if args.batch:
            cmd = cmd.strip() + (" &&" if module != list(pipeline)[-1] else "")
            srun_args = (
                f"--{k.replace('_', '-')}={v}" for k, v in slurm_kwargs.items() if v is not None
            )
            slurm.add_cmd(" ".join([srun_cmd, *srun_args, cmd]))
        else:
            t0 = time.time()
            ret = (
                slurm.srun(cmd, srun_cmd=srun_cmd)
                if slurm
                else subprocess.run(cmd, shell=True, check=True).returncode
            )
            if ret:
                logging.error(f"Running {module} fails ! Check previous errors.")
            else:
                logging.info(
                    f"Module '{module}' succesfully runs in {parse_time(time.time() - t0)}"
                )
                params.update(
                    slurm=slurm_kwargs,
                    done=datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                    duration=parse_time(time.time() - t0, human=False),
                )
                with open(updated_yaml_file, "w") as f:
                    yaml.dump(updated_pipeline_dict, f, sort_keys=False)

    if args.test:
        logging.info(
            "Successful test of the pipeline. You can now run it without the '--test' option"
        )
        return

    if args.batch:
        sbatch_kwargs = dict(shell="/bin/bash")
        logging.info(
            "The following script will be sent to slurm batch system:\n"
            + slurm.script(**sbatch_kwargs)
        )
        if input("Do you want to proceed [y/N] ? ") not in list("yY"):
            return
        job_id = slurm.sbatch(postcmd, **sbatch_kwargs)
        logging.info(f"Job '{job_id}' has been sent.")
    else:
        # Print info time
        info, total_time = "", 0.0
        for module, params in updated_pipeline_dict.get("pipeline", {}).items():
            duration = params.get("duration")
            slurm = params.get("slurm") or dict()
            ntasks, cpus_per_task = slurm.get("ntasks", 1), slurm.get("cpus_per_task", 256)
            info += f"\n - '{module}' takes {parse_time(duration)} "
            info += f"({ntasks} task" + plural(ntasks)
            info += f" with {cpus_per_task} cpu" + plural(cpus_per_task) + " per task "
            info += f"- {params.get('done')})"
            h, m, s = map(int, duration.split(":"))
            total_time += datetime.timedelta(hours=h, minutes=m, seconds=s).seconds
        logging.info(
            f"Pipeline runs in {parse_time(total_time)} with the following amount of time per module:"
            + info
        )


if __name__ == "__main__":
    main()
