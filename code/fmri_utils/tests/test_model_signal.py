""" Test script for model_signal module

Test with ``py.test test_model_signal.py``.
"""

from os.path import dirname, join as pjoin

MY_DIR = dirname(__file__)
EXAMPLE_FILENAME = 'ds107_sub012_t1r2_small.nii'
EXAMPLE_FULLPATH = pjoin(MY_DIR, EXAMPLE_FILENAME)

import numpy as np
import scipy.stats as sst
from scipy.stats import gamma
import nibabel as nib

# This import needs the code directory on the Python PATH
from fmri_utils import data_timecourse, create_design_matrix, \
     event_timecourse, hrf, b_e_calc, create_contrast_img

# Imports for testing
from numpy.testing import assert_almost_equal

def test_data_timecourse():
    # get data
    img = nib.load(EXAMPLE_FULLPATH)
    data = img.get_data()
    # get random coordinates and number of dummy frames
    coord = [np.random.randint(data.shape[i]) for i in range(3)]
    d_n = np.random.randint(data.shape[-1])
    # remove dummy frames
    data = data[..., d_n:]
    # get timecourse from coord manually
    t_c0 = data[coord[0],coord[1],coord[2]]
    # get timecourse from data_timecourse
    t_c1 = data_timecourse(EXAMPLE_FULLPATH, coord, [], d_n)
    # assert same for coordinates
    assert np.allclose(t_c0, t_c1, rtol=1e-4)
    # create random mask
    mask = np.random.randint(2,size=data.shape[:-1]).astype(bool)
    # get timecourse for mask
    t_c2 = data[mask].T
    t_c3 = data_timecourse(EXAMPLE_FULLPATH, [], mask, d_n)
    # assert same for masks
    assert np.allclose(t_c2, t_c3, rtol=1e-4)
    # get timecourse for all voxels
    t_c4 = np.reshape(data, (np.prod(data.shape[:-1]), data.shape[-1])).T
    t_c5 = data_timecourse(EXAMPLE_FULLPATH, [], [], d_n)
    # assert same for all voxels
    assert np.allclose(t_c4, t_c5, rtol=1e-4)

def test_create_design_matrix():
    # create random number trs, tr, and dummy frames
    n_tr = np.random.randint(100)
    tr = np.random.randint(4) + np.random.rand(1)
    d_n = np.random.randint(n_tr)
    # create random timecourse of 0s and 1s with random n of columns
    col_n = np.random.randint(10)
    t_c = np.random.randint(2,size=(n_tr, col_n))
    # create design matrix manually
    des0 = np.ones((n_tr - d_n, t_c.shape[1]+1))
    hrf_at_trs = hrf(np.arange(0, 30, tr))
    n_to_remove = len(hrf_at_trs) - 1
    for i in range(t_c.shape[1]):
        convolved = np.convolve(t_c[:,i], hrf_at_trs)
        des0[:,i] = convolved[d_n:-n_to_remove]
    # create design matrix from module
    des1 = create_design_matrix(t_c, tr, n_tr, d_n)
    # assert same
    assert np.allclose(des0, des1, rtol=1e-4)

def test_event_timecourse():
    assert True

def test_hrf():
    # create random tr and set of times
    tr = np.random.randint(4) + np.random.rand(1)
    times = np.arange(0, np.random.randint(100), tr)
    # create peak values and undershoot values
    peak_values = gamma.pdf(times, 6)
    undershoot_values = gamma.pdf(times, 12)
    # combine and scale to .6
    values0 = peak_values - 0.35 * undershoot_values
    values0 = values0 / np.max(values0) * 0.6
    # get values from hrf
    values1 = hrf(times)
    # assert same values
    assert np.allclose(values0, values1)

def test_b_e_calc():
    # get random data
    X0 = np.random.randn(100)
    Y0 = np.random.randn(100)
    # get beta values from scipy stats
    B0 = np.ones(2)
    B0[0], B0[1], _, _, _ = sst.linregress(X0, Y0)
    # get beta values and residuals using b_e_calc
    X1 = np.ones((100,2))
    X1[:,0] = X0
    Y1 = Y0
    B1, e1 = b_e_calc(Y1, X1)
    # calculate residuals
    e0 = Y1 - X1.dot(B0)
    # assert same betas and residuals
    assert np.allclose(B0, B1)
    assert np.allclose(e0, e1)

def test_create_contrast_img():
    # create random beta maps
    vol_shape = np.random.randint(100, size=(3,))
    B = np.random.randn(2, np.prod(vol_shape))
    C = np.array([0, 1])
    # calculate contrast map
    Bmap = C.T.dot(B)
    Cmap0 = np.reshape(Bmap, (vol_shape))
    # get cmap from create_contrast_img
    Cmap1 = create_contrast_img(B, C, vol_shape)
    # assert same maps
    assert np.allclose(Cmap0, Cmap1)
