"""
Code implementing algorithms in SPM
"""
# Python 2 compatibility
from __future__ import print_function, division

import numpy as np

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
