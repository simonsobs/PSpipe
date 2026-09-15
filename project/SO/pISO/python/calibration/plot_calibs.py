import yaml
import numpy as np
import pylab as plt
import argparse
import os
from pspy import so_dict


parser = argparse.ArgumentParser()
parser.add_argument('odir', help="Where to save figures")
parser.add_argument('--calib-yamls', nargs="+",
                    help='A list of calib resutls yamls fils to plot')
parser.add_argument('--paramfiles', nargs="+",
                    help='A list of paramfiles to plot calibs (overrides --calib-yamls)')
parser.add_argument('--arrays', nargs="+",
                    help='list of arrays to plot (None = all LAT ISO)', default=None)
parser.add_argument('--names', nargs="+", help='optionnally give names to these calibs', default=None)
parser.add_argument('--colors', nargs="+", help='optionnally give colors to these calibs', default=None)
parser.add_argument('--test', help='Which test to use', default='_all_tubes_Pl143')
parser.add_argument('--ylabel', help='Ylabel for the figure', default='Calibration Factor')
parser.add_argument('--plot-line', help='axhline for figure, is also the value to compare to get nsigmas', default=1, type=float)
parser.add_argument('--plot-mean', help='plot mean of all data points (ASSUME UNCORRELATED POINTS)', default=False)


args = parser.parse_args()

test_names = args.names or [os.path.basename(os.path.dirname(yaml_fn)) for yaml_fn in args.calib_yamls]
test_colors = args.colors or [f"C{c}" for c, yaml_fn in enumerate(args.calib_yamls)]

defaults_calibs = {
    "i1_f090":1,
    "i1_f150":1,
    "i3_f090":1,
    "i3_f150":1,
    "i4_f090":1,
    "i4_f150":1,
    "i6_f090":1,
    "i6_f150":1,
    "c1_f220":1,
    "c1_f280":1,
    "i5_f220":1,
    "i5_f280":1,
}

if args.paramfiles is None:
    calib_yamls = args.calib_yamls
    defaults = None
else:
    defaults = {}
    calib_yamls = []
    for name, prmfl in zip(args.names, args.paramfiles):
        d = so_dict.so_dict()
        d.read_from_file(prmfl)
        calib_yamls.append(f"calib/{d['run_name']}/calibs_dict{args.test}.yaml")

results_dict = {}
errs_dict = {}
for name, yaml_fn in zip(args.names, calib_yamls):
    with open(yaml_fn, 'r') as f:
        results_dict[name] = yaml.safe_load(f)
    with open(yaml_fn.replace('_dict', '_errs_dict'), 'r') as f:
        errs_dict[name] = yaml.safe_load(f)

if args.paramfiles is not None:
    for name, prmfl in zip(args.names, args.paramfiles):
        d = so_dict.so_dict()
        d.read_from_file(prmfl)
        defaults[name] = {ar[-7:]: d[f"cal_{ar}"] for ar in results_dict[name].keys()}

arrays_set = args.arrays or ['c1_f220', 'c1_f280', 'i1_f090', 'i1_f150', 'i3_f090', 'i3_f150', 'i4_f090', 'i4_f150', 'i5_f220', 'i5_f280', 'i6_f090', 'i6_f150']

fig, ax = plt.subplots(figsize=(12, 6))
ax.axhline(args.plot_line, color='grey', ls='--', lw=1)

x_shift = np.linspace(-.02*len(test_names), .02*len(test_names), len(test_names))

for j, (test, results_subdict) in enumerate(results_dict.items()):
    for i, (name, cal) in enumerate(results_subdict.items()):
        std = errs_dict[test][name]
        ax.errorbar(
            i + x_shift[j],
            cal,
            std,
            color=test_colors[j],
            marker=".",
            ls="",
            label=test if i == 0 else None,
            markersize=6.5,
            # markeredgewidth=2,
            fillstyle='none',
        )
        if defaults is not None:
            ax.plot(
                i + x_shift[j],
                defaults[test][arrays_set[i]],
                color=test_colors[j],
                marker="_",
                ls="",
                markersize=6.5,
                alpha=1,
                markeredgewidth=2,
            )
    if args.plot_mean:
        mean_var = 1 / np.sum([1 / errs_dict[test][name]**2 for name in results_subdict.keys()])
        mean_value = np.sum([results_subdict[name] / errs_dict[test][name]**2 for name in results_subdict.keys()]) * mean_var
        ax.errorbar(
                i+1 + x_shift[j],
                mean_value,
                np.sqrt(mean_var),
                color=test_colors[j],
                marker=".",
                ls="",
                markersize=6.5,
                # markeredgewidth=2,
                fillstyle='none',
            )
        print(f"{test}: {mean_value=:.5f}±{np.sqrt(mean_var):.5f}")
# ax.set_ylim(.79, 1.02)
ax.legend(fontsize=15)

xlabels = [ar[-7:] for ar in arrays_set]
if args.plot_mean:
    xlabels += ["Mean"]
x = np.arange(0, len(xlabels))
ax.set_xticks(x, xlabels)
ax.set_ylabel(args.ylabel, fontsize=18)
plt.tight_layout()
plt.savefig(f"{args.odir}/{args.ylabel.replace(" ", "_")}_summary_{'_'.join(test_names[:len(results_dict.keys())])}.pdf", bbox_inches="tight")
plt.clf()
plt.close()

# Claude wrote that
def print_table(headers, rows, title=None):
    # Column widths: max of header or any cell value
    col_widths = [
        max(len(str(headers[i])), max((len(str(row[i])) for row in rows), default=0))
        for i in range(len(headers))
    ]

    # Box-drawing pieces
    top    = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    mid    = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    bottom = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"

    total_width = sum(w + 3 for w in col_widths) + 1

    def row_line(cells, widths):
        return "│" + "│".join(f" {str(c):<{w}} " for c, w in zip(cells, widths)) + "│"

    # Print
    if title:
        print("┌" + "─" * (total_width - 2) + "┐")
        print("│" + title.center(total_width - 2) + "│")

    print(top)
    print(row_line(headers, col_widths))
    print(mid)
    for row in rows:
        print(row_line(row, col_widths))
    print(bottom)

for j, (test, results_subdict) in enumerate(results_dict.items()):
    for i, (name, cal) in enumerate(results_subdict.items()):
        std = errs_dict[test][name]

headers = ["params"] + [test for test in results_dict.keys()]
rows = [[name] + [f"{results_dict[test][name]:.4f}±{errs_dict[test][name]:.4f}   ({(results_dict[test][name] - args.plot_line) / errs_dict[test][name]:.1f}σ)" for test in results_dict.keys()] for name in results_dict[test].keys()]

print_table(headers, rows, title=args.ylabel)