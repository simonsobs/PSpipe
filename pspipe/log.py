import logging


def get_logger(fmt=None, datefmt=None, debug=False, **kwargs):
    """Return logger from logging module

    Parameters
    ----------

    fmt: string
      the format string that preceeds any logging message
    datefmt: string
      the date format string
    debug: bool
      debug flag
    """
    fmt = fmt or "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s"
    datefmt = datefmt or "%d-%b-%y %H:%M:%S"
    logging.basicConfig(
        format=fmt, datefmt=datefmt, level=logging.DEBUG if debug else logging.INFO, force=True
    )
    return logging.getLogger(kwargs.get("name"))
