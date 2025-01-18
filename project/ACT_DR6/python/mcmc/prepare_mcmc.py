import argparse
import os
import shutil

from cobaya.install import install_script
from pspipe_utils.log import get_logger


def main(args=None):
    parser = argparse.ArgumentParser(description="A python pipe for CMB analysis")
    parser.add_argument(
        "-d", "--debug", help="Enable debug level", default=False, action="store_true"
    )
    parser.add_argument(
        "-s",
        "--sacc-file",
        help="Set the sacc file location",
        type=os.path.realpath,
        default="sacc/dr6_data_sacc.fits",
    )
    parser.add_argument("--run", help="Set MCMC run name", default="DR6base", type=str)
    args = parser.parse_args(args)

    log = get_logger(debug=args.debug)

    if args.run not in (supported_runs := ["DR6base"]):
        logging.error(
            f"MCMC run '{args.run}' not supported! "
            + f"Only '{supported_runs}' currently supported."
        )
        raise SystemExit()

    # MCMC output directory
    mcmc_dir = os.path.realpath("mcmc")

    # Let's create the packages for cobaya without running cobaya-install and copy sacc file
    # produced by the DR6 pipeline
    mflike_data_folder = "ACTDR6MFLike/trunk"
    packages_path = os.path.join(mcmc_dir, "packages")
    mflike_full_data_folder = os.path.join(packages_path, "data", mflike_data_folder)
    os.makedirs(mflike_full_data_folder, exist_ok=True)
    log.info(f"Copying '{args.sacc_file}' into '{mflike_full_data_folder}' directory")
    shutil.copy2(args.sacc_file, mflike_full_data_folder)

    # Read yaml template for cobaya
    this_directory = os.path.realpath(os.path.dirname(__file__))
    with open(os.path.join(this_directory, f"{args.run}.yaml.tmpl"), "r") as f:
        yaml_tmpl = f.read()
    # Format does not work here since there are a lot LaTeX expression with {} delimiters
    kwargs = dict(
        data_folder=mflike_data_folder,
        input_file=os.path.basename(args.sacc_file),
        covmat=os.path.join(this_directory, f"{args.run}_initial.covmat"),
        packages_path=os.path.realpath(packages_path),
    )
    for field, value in kwargs.items():
        yaml_tmpl = yaml_tmpl.replace(f"@@{field}@@", value)

    with open(yaml := os.path.join(mcmc_dir, f"{args.run}.yaml"), "w") as f:
        f.write(yaml_tmpl)
    log.info(f"Cobaya yaml file '{yaml}' generated.")

    # Now install likelihood data via cobaya (no code installation to avoid cmb solver overloading)
    install_script(f"--just-data {yaml}".split())


if __name__ == "__main__":
    main()
