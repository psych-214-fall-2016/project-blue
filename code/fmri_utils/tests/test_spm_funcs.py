""" Test script for spm_funcs module

Test with ``py.test test_spm_funcs.py``.
"""

from os.path import dirname, join as pjoin

MY_DIR = dirname(__file__)
EXAMPLE_FILENAME = 'ds107_sub012_t1r2_small.nii'

import numpy as np

import nibabel as nib

# This import needs the code directory on the Python PATH
from fmri_utils import get_spm_globals, spm_global, spm_hrf

# Imports for testing
from numpy.testing import assert_almost_equal


def test_spm_globals():
    # Test get_spm_globals and spm_global functions
    example_path = pjoin(MY_DIR, EXAMPLE_FILENAME)
    expected_values = np.loadtxt(pjoin(MY_DIR, 'global_signals.txt'))
    glob_vals = get_spm_globals(example_path)
    assert np.allclose(glob_vals, expected_values, rtol=1e-4)
    img = nib.load(example_path)
    data = img.get_data()
    globals = []
    for vol_no in range(data.shape[-1]):
        vol = data[..., vol_no]
        globals.append(spm_global(vol))
    assert np.allclose(globals, expected_values, rtol=1e-4)


def test_spm_hrf():
    # Test calculation of SPM HRF
    # Test our code against saved output from SPM
    # For a list of time gaps (gaps between time samples)
    for dt_str in ('0.1', '1', '2.5'):
        # Convert string e.g. 0.1 to string e.g. '0p1'
        dt_p_for_point = dt_str.replace('.', 'p')
        # Load saved values from SPM for this time difference
        spm_fname = 'spm_hrf_' + dt_p_for_point + '.txt'
        spm_values = np.loadtxt(pjoin(MY_DIR, spm_fname))
        # Time difference as floating point value
        dt = float(dt_str)
        # Make times corresponding to samples from SPM
        times = np.arange(len(spm_values)) * dt
        # Evaluate our version of the function at these times
        our_values = spm_hrf(times)
        # Our values and the values from SPM should be about the same
        assert_almost_equal(spm_values, our_values)
