import matplotlib

matplotlib.use("Agg")
import argparse
import os

from cobaya.yaml import yaml_load_file
from getdist import loadMCSamples
from getdist.plots import get_subplot_plotter
from getdist.types import ResultTable
from pspipe_utils.log import get_logger


def get_plot_settings(settings=None):
    from getdist.plots import GetDistPlotSettings

    plot_settings = GetDistPlotSettings()
    unkown_settings = dict()

    settings = settings or dict()
    for name, value in settings.items():
        if hasattr(plot_settings, name):
            setattr(plot_settings, name, value)
        else:
            unkown_settings.setdefault(name, value)

    return dict(settings=plot_settings, **unkown_settings)


def main(args=None):
    parser = argparse.ArgumentParser(description="A python pipe for CMB analysis")
    parser.add_argument(
        "-d", "--debug", help="Enable debug level", default=False, action="store_true"
    )
    parser.add_argument("--getdist-yaml", help="Path to a yaml configuration for getdist")
    args = parser.parse_args(args)

    log = get_logger(debug=args.debug)

    # Getdist configuration if any
    config = yaml_load_file(args.getdist_yaml) if args.getdist_yaml else dict()

    # Plot directory
    plot_dir = os.path.realpath(config.get("plot_dir", "plots/mcmc"))
    os.makedirs(plot_dir, exist_ok=True)
    log.info(f"Figures/tables will be stored in '{plot_dir}'")

    # Build samples dict
    samples = {}
    for name, meta in config.get("samples", {}).items():
        log.info(f"Loading '{name}' samples...")
        if not (path := meta.get("path")):
            logging.error("Missing path to samples!")
            raise SystemExit()
        samples.setdefault(name, {}).update(
            samples=loadMCSamples(
                path,
                no_cache=config.get("no_cache", meta.pop("no_cache", False)),
                settings={"ignore_rows": config.get("burnin", meta.pop("burnin", 0.3))},
            ),
            **meta,
        )

    if config.get("plot_progress", True):
        from cobaya.samplers.mcmc import plot_progress

        info = "Gelman-Rubin values:"
        for name, meta in samples.items():
            axes = plot_progress(meta.get("path"))
            axes[0].get_figure().savefig(filename := os.path.join(plot_dir, f"{name}_progress.pdf"))
            log.info(f"Progress status stored in '{filename}'")
            info += (
                f"\n - {name} ({meta.get('label')}): R-1 = {meta.get('samples').getGelmanRubin()}"
            )

        log.info(info)

    # Get default GetDist settings
    default_settings = get_plot_settings(config.get("settings"))
    log.debug(default_settings)

    # Build plots
    for name, plot in config.get("plots", {}).items():
        log.info(f"Generating '{name}' plot...")

        gdplot = get_subplot_plotter(**(default_settings | plot.get("settings", {})))
        if not (kind := plot.get("kind")):
            log.error("Missing 'kind' parameter!")
            raise SystemExit()
        if not (plot_function := getattr(gdplot, kind)):
            log.error("Unkown '{kind}' plot function! Check getdist documentation.")
            raise SystemExit()

        # Uses aliases for parameters (if any)
        default_parameters = config.get("parameters", {})
        params = plot.get("params")
        if isinstance(params, str):
            plot.update(params=default_parameters.get(params, params))
        elif isinstance(params, (list, tuple)):
            new_params = []
            for par in params:
                new_params += [default_parameters.get(par, [par])]
            plot.update(params=sum(new_params, []))

        # Build kwargs
        _get = lambda field: [samples[samp].get(field) for samp in plot.get("samples", [])]
        if labels := plot.get("legend_labels"):
            if isinstance(labels, str):
                plot.update(legend_labels=[labels])
        else:
            plot.update(legend_labels=_get("label"))

        if colors := plot.get("colors"):
            if isinstance(colors, str):
                plot.update(colors=[colors])
        else:
            plot.update(colors=_get("color"))

        # Fixes for getdist
        if not "contour_colors" in plot:
            plot.update(contour_colors=plot.get("colors"))

        # GetDist plot
        log.debug(plot)
        plot_function(_get("samples"), **plot)
        gdplot.export(
            filename := plot.get("filename", os.path.join(plot_dir, f"{kind}_{name}.pdf"))
        )
        log.info(f"Export '{kind}' to '{filename}'")

        # GetDist table
        if table := plot.get("table"):
            log.debug(table)
            log.info(f"Generating '{name}' table...")
            t = ResultTable(
                results=_get("samples"), titles=_get("label"), paramList=plot.get("params"), **table
            )
            t.tablePNG(
                filename=(
                    filename := table.get("filename", os.path.join(plot_dir, f"table_{name}.png"))
                ),
                dpi=300,
            )
            t.write(filename.replace(".png", ".tex"))
            log.info(f"Export table results to '{filename}'")


if __name__ == "__main__":
    main()
