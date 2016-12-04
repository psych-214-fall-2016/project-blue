""" Test script for detectors module

Test with ``py.test test_detectors.py``.
"""

from os.path import dirname, join as pjoin

MY_DIR = dirname(__file__)
EXAMPLE_FILENAME = 'ds107_sub012_t1r2_small.nii'

import numpy as np
import numpy.linalg as npl
import nibabel as nib

# This import needs the code directory on the Python PATH
from fmri_utils import mean_detector, std_detector, iqr_detector

# Imports for testing
from numpy.testing import assert_almost_equal

def test_mean_detector():
    # create data of ones and outliers to set large means
    data = np.ones((64,64,30,100))
    outliers0 = np.random.randint(100,size=(7,))
    outliers0 = np.sort(outliers0)
    # set trs to large standard deviations
    data[...,outliers0] = 100
    # get standard deviations for each volume
    vol_mean0 = []
    for i in range(data.shape[-1]):
        vol_mean0.append(data[...,i].mean())
    # get standard deviations and outliers from std_detector
    vol_mean1, outliers1 = mean_detector(data)
    # assert same
    assert np.allclose(vol_mean0, vol_mean1)
    assert np.allclose(outliers0, outliers1)

def test_std_detector():
    # create data of ones and outliers to set large standard deviations
    data = np.ones((64,64,30,100))
    outliers0 = np.random.randint(100,size=(7,))
    outliers0 = np.sort(outliers0)
    # set trs to large standard deviations
    data[:,:,0:15,outliers0] = -100
    data[:,:,15:,outliers0] = 100
    # get standard deviations for each volume
    vol_std0 = []
    for i in range(data.shape[-1]):
        vol_std0.append(data[...,i].std())
    # get standard deviations and outliers from std_detector
    vol_std1, outliers1 = std_detector(data)
    # assert same
    assert np.allclose(vol_std0, vol_std1)
    assert np.allclose(outliers0, outliers1)

def test_iqr_detector():
    # create random data and IQR factor
    data = np.random.randn(10)
    iqr_factor = np.random.randint(3)
    # get quartile range
    q1, q3 = np.percentile(data, [25,75])
    iqr = q3 - q1
    upper_lim = q3 + iqr * iqr_factor
    lower_lim = q1 - iqr * iqr_factor
    # find outliers
    outlier_tf0 = np.logical_or(data > upper_lim, data < lower_lim)
    outlier_i0 = [i for i, x in enumerate(outlier_tf0) if x]
    # find outliers using iqr_detector
    outlier_tf1, outlier_i1 = iqr_detector(data, iqr_factor)
    # asser same
    assert np.allclose(outlier_tf0, outlier_tf1)
    assert np.allclose(outlier_i0, outlier_i1)
