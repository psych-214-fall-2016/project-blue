""" Test script for slice_timing_corr module

Test with ``py.test test_slice_timing_corr.py``.
"""

import numpy as np
from fmri_utils import slice_timing_corr

def test_slice_timing_corr():
    """
    Test creates fake data (using real TR and time offsets) and runs to make sure the code does not crash
    """
    I, J, K, T = 2, 3, 4, 30
    Y = np.random.normal(5, 1, size=(I, J, K, T))
    TR = 2
    time_offsets = [0.08, 1.08, 0.15, 1.15, 0.23, 1.23, 0.31, 1.31, 0.38, 1.38, 0.46, 1.46, 0.54, 1.54, 0.62, 1.62, 0.69, 1.69, 0.77, 1.77, 0.85, 1.85, 0.92, 1.92, 1.00, 2.00]

    slice_timing_corr(Y, TR, time_offsets)

    return
