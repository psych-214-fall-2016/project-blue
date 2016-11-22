import numpy as np
from fmri_utils import slice_timing_corr

def test_slice_timing_corr():
    """
    Test should create 4D fake data and put through the slice timing code

    Currently, creates fake data and runs to make sure the code does not crash
    But would be better if it somehow tested the reslicing was working...
    """

    I, J, K, T = 2, 3, 4, 30
    Y = np.random.normal(5, 1, size=(I, J, K, T))
    TR = 2.5

    slice_timing_corr.slice_timing_corr(Y, TR)

    #what would this test look like...?

    return
