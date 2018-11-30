"""
@brief: a module to handle base configuration
"""

from __future__ import print_function
import os, argparse

PSPIPE_ROOT           = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
DEFAULT_OUTPUT_DIR   = os.path.join(PSPIPE_ROOT, "output")
DEFAULT_DATA_DIR = os.path.join(PSPIPE_ROOT, "resource")

argparser = argparse.ArgumentParser(
    prog='pspipe',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    conflict_handler='resolve')


def get_output_dir():
    ''' return default output directory '''
    return DEFAULT_OUTPUT_DIR

def get_data_dir():
    ''' return default data directory '''
    return DEFAULT_DATA_DIR

