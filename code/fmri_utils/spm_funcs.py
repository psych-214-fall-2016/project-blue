"""
Code implementing algorithms in SPM

The functions have docstrings according to the numpy docstring standard - see:

    https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
"""
# Python 2 compatibility
from __future__ import print_function, division

import numpy as np
from scipy.stats import gamma

import nibabel as nib


def spm_global(vol):
    """ Calculate SPM global metric for array `vol`

    Parameters
    ----------
    vol : array
        Array giving image data, usually 3D.

    Returns
    -------
    g : float
        SPM global metric for `vol`
    """
    T = np.mean(vol) / 8
    return np.mean(vol[vol > T])


def get_spm_globals(fname):
    """ Calculate SPM global metrics for volumes in image filename `fname`

    Parameters
    ----------
    fname : str
        Filename of file containing 4D image

    Returns
    -------
    spm_vals : array
        SPM global metric for each 3D volume in the 4D image.
    """
    img = nib.load(fname)
    data = img.get_data()
    spm_vals = []
    for i in range(data.shape[-1]):
        vol = data[..., i]
        spm_vals.append(spm_global(vol))
    return spm_vals


def spm_hrf(times):
    """ Return values for standard SPM HRF at given `times`

    This is the same as SPM's ``spm_hrf.m`` function using the default input
    values.

    Parameters
    ----------
    times : array
        Times at which to sample hemodynamic response function

    Returns
    -------
    values : array
        Array of same length as `times` giving HRF samples at corresponding
        time post onset (where onset is T==0).
    """
    # Gamma only defined for x values > 0
    time_gt_0 = times > 0
    ok_times = times[time_gt_0]
    # Output vector
    values = np.zeros(len(times))
    # Gamma pdf for the peak
    peak_values = gamma.pdf(ok_times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(ok_times, 16)
    # Combine them
    values[time_gt_0] = peak_values - undershoot_values / 6.
    # Divide by sum
    return values / np.sum(values)
