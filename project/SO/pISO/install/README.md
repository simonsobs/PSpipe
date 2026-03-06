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

### first do
this