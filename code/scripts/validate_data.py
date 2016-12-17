"""Validate data downloaded from openfmri

This script will run the dir_utils module's validate_data function

Run as scripts/validate_data.py data
"""

# import validate_data
import sys
from fmri_utils import validate_data

def run_validation(data_dir):
    # run validate_data
    validate_data(data_dir)

if __name__ == '__main__':
    # if not enough arguments, raise error
    if len(sys.argv) < 2:
        raise ValueError('Please enter the data directory')

    # get data_dir from inputs
    data_dir = sys.argv[1]

    # run validate_data
    run_validation(data_dir)
