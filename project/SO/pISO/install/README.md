These are basic instructions to create environments that run PSpipe on various clusters.

# tiger
When we say "environment", its more like a "project". Part of it is the literal 
virtual python environment for code to run, but it's also about how to structure
your code locally. Assuming your project is named `PSpipe_dev` (for example), and you put your projects in a directory like `~/projects`, these instructions will result in a structure like this:
```
zatkins@tiger3:~/projects/PSpipe_dev$ ll
total 56
drwxr-xr-x.  6 zatkins physics   140 Mar  6 10:20 ./
drwxr-xr-x.  4 zatkins physics    48 Mar  6 11:20 ../
lrwxrwxrwx.  1 zatkins physics    18 Mar  6 10:19 activate -> .venv/bin/activate
drwxr-xr-x.  3 zatkins physics  4096 Mar  6 10:16 install/
drwxr-xr-x. 10 zatkins physics   157 Mar  6 10:19 repos/
drwxr-xr-x.  2 zatkins physics 28672 Mar  6 11:20 slurm_output/
drwxr-xr-x.  7 zatkins physics  4096 Mar  6 10:19 .venv/
```
where `activate` is a handy symlink to the venv, `install` is what we will download 
from `PSpipe` here, `repos` are where local packages that `PSpipe` needs will live,
`slurm_output` is a location for slurm output files (good for record-keeping when
you want to review job performance), and of course `.venv` is where the actual
python environment (named `PSpipe_dev`) lives.

If you are happy with this setup, please continue reading. Note, of course you
can change where your project will live (it doesn't have to be in `~/projects`)
and what it's called (it doesn't have to be `PSpipe_dev`), but for now we just
use this as an example.

### install
Download `install.sh` into your project's `install` directory:
```
mkdir -p ~/projects/PSpipe_dev/install
cd ~/projects/PSpipe_dev/install
wget --no-cache https://raw.githubusercontent.com/simonsobs/PSpipe/zach_piso/project/SO/pISO/install/install.sh
```

All we need to do is execute this file in bash, but before doing that, please
read to understand what it will do. First, there is a mandatory argument, which
is `true` or `false`: whether to compile `enlib/array_ops`. You say `true` if 
you will build `mnms` noise models, otherwise `false`. Then, if you want your
slurm jobs to send you an email, add your email (this is a good idea).

1. It downloads all the other files next to `install.sh` to the current directory.

2. It creates a module file with needed libraries both for installation and
runtime. The file is `tiger_module_250723`. It will copy this file to your 
personal Modules space as `~/Modules/modulefiles/tiger3/250723`. It will also 
populate the `_ENLIB_PATH` (see `enlib` below). Finally, it activates this 
Module.

3. It clones the repositories in `local_requirements.txt` to `../repos` and 
switches them to the correct branches. `enlib` is moved into an intermediate 
directory, `../repos/_enlib`, so that in the Module file, we can add this 
intermediate `_enlib` directory to the `PYTHONPATH` rather than the entire `repos`
directory. (This is because `enlib` must be "installed" by adding to the `PYTHONPATH`, 
but the package is also a module. The intermediate `_enlib` directory is a hacky
way to turn it into the `_enlib` package containing the `enlib` module).

4. It creates a python 3.12 venv with the name `PSpipe_dev` (or whatever you
have called your project). It installs all packages:
* First `numpy<2` which should be `numpy 1.26.4` (for `ducc`)
* Then `ducc` which it compiles from source from `../repos/ducc` (for `pixell`)
* Then everything in `requirements.in` (including `pixell`)
* Then all the other local repositories (other than `ducc`):
    * NOTE: By default `enlib` is just moved into `../repos/_enlib` and added to
    the `PYTHONPATH` via the module file. If you are also building `mnms` noise
    models, you want to pass `true` as a command-line argument to `install.sh`,
    see above. This will compile the `array_ops` module within `enlib`, with some
    changes made to `compile_opts`, see `enlib_array_ops_python3.12_Makefile`.
    * NOTE: `PSpipe` is just copied in step 3: it's just a collection of pure
    python scripts so it doesn't need to be "installed"
* It creates a link from `../.venv/bin/activate` to `../activate`, so you can use
`source ~/projects/PSpipe_dev/activate` as a shortcut to activate the venv.

5. It modifies the `tiger.slurm` file at `../repos/PSpipe/project/SO/pISO/slurm/tiger.slurm`
so that it uses the slurm output directory and venv specific to your user. It 
also adds slurm email lines if you provided your email.

If you are happy with that, you can install via:
```
sh install.sh [true/false] [optional: email]
```

You may want to add the lines
```
module purge
module load tiger3/250723
```
to your ~/.bashrc so that the modules are loaded as soon as you log in.