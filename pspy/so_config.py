"""
@brief: a module to handle base configuration
"""

from __future__ import absolute_import, print_function
import os, argparse
import configparser

PSPIPE_ROOT        = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
DEFAULT_OUTPUT_DIR = os.path.join(PSPIPE_ROOT, "output")
DEFAULT_DATA_DIR   = os.path.join(PSPIPE_ROOT, "data")
DEFAULT_CONFIG_DIR = os.path.join(PSPIPE_ROOT, "configs")

argparser = argparse.ArgumentParser(
    prog='pspipe',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    conflict_handler='resolve')

configparser = configparser.SafeConfigParser()
configparser.optionxform = str

def get_output_dir():
    ''' return default output directory '''
    return DEFAULT_OUTPUT_DIR

def get_data_dir():
    ''' return default data directory '''
    return DEFAULT_DATA_DIR

def load_config(config_file):
    global configparser

    path_to_config = os.path.join(DEFAULT_CONFIG_DIR, config_file)
    configparser.read(path_to_config)


