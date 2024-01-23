
import numpy as np
import pspipe
import os
from itertools import combinations_with_replacement as cwr

multistep_path = os.path.join(os.path.dirname(pspipe.__file__), "js")

arrays = ["pa4_f220", "pa5_f090", "pa5_f150", "pa6_f090", "pa6_f150"]
spectra = ["TT", "TE", "ET", "TB", "BT", "EE", "EB", "BE", "BB"]

null_list = []
for i, (el1, el2) in enumerate(cwr(arrays, 2)):
    for j, (el3, el4) in enumerate(cwr(arrays, 2)):
        if j <= i: continue
        null_list += [(el1, el2, el3, el4)]

os.system(f"cp {multistep_path}/multistep2.js .")
filename = f"array_null.html"
g = open(filename, mode='w')
g.write('<html>\n')
g.write('<head>\n')
g.write('<title> array null </title>\n')
g.write('<script src="multistep2.js"></script>\n')
g.write('<script> add_step("null", ["c","v"]) </script> \n')
g.write('<script> add_step("spec", ["j","k"]) </script> \n')
g.write('<script> add_step("array", ["a","z"]) </script> \n')
g.write('</head> \n')
g.write('<body> \n')
g.write('<h1>array null test </h1>')
g.write('<p> In this webpage we host all plots for array null tests, we have done 181 tests corresponding to the following  PTE distribution </p>')
g.write('<img src="' + 'plots/array_nulls/pte_hist_all_spectra_corrected+mc_cov+beam_cov+leakage_covskip_pa4pol_skip_EB_skip_TT_diff_freq.png' + '" width="50%" /> \n')
g.write('<p> you can have a look at all spectra, press c/v to change the null, a/z to change spectrum (TT,TE, .. BB) </p>')


g.write('<div class=null> \n')

for null in null_list:

    el1, el2, el3, el4 = null
    g.write('<div class=array>\n')


    for spec in spectra:
        str = f"plots/array_nulls/diff_{spec}_dr6_{el1}xdr6_{el2}_dr6_{el3}xdr6_{el4}.png"
        g.write('<img src="' + str + '" width="50%" /> \n')
    g.write('</div>\n')

g.write('</div> \n')
g.write('</body> \n')
g.write('</html> \n')
g.close()

