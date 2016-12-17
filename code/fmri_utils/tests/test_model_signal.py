""" Test script for model_signal module

Test with ``py.test test_model_signal.py``.
"""

from os.path import dirname, join as pjoin

MY_DIR = dirname(__file__)
EXAMPLE_FILENAME = 'ds107_sub012_t1r2_small.nii'
EXAMPLE_FULLPATH = pjoin(MY_DIR, EXAMPLE_FILENAME)

import numpy as np
import numpy.linalg as npl
import scipy.stats as sst
from scipy.stats import gamma
import nibabel as nib
from skimage.filters import threshold_otsu

# This import needs the code directory on the Python PATH
from fmri_utils import data_timecourse, create_design_matrix, \
     event_timecourse, poly_drift, beta_res_calc, compute_tstats, spm_hrf

def test_data_timecourse():
    # get data
    img = nib.load(EXAMPLE_FULLPATH)
    data = img.get_data()
    # get random coordinates
    coord = [np.random.randint(data.shape[i]) for i in range(3)]
    mask0 = np.zeros(data.shape[:-1]).astype(bool)
    mask0[coord[0], coord[1], coord[2]] = True
    # get timecourse from coord manually
    t_c0 = data[mask0].T
    # get timecourse from data_timecourse
    t_c1, mask1 = data_timecourse(data, coord, [])
    # assert same for coordinates
    assert np.allclose(t_c0, t_c1, rtol=1e-4)
    assert np.allclose(mask0, mask1, rtol=1e-4)
    # create random mask
    mask0 = np.random.randint(2,size=data.shape[:-1]).astype(bool)
    # get timecourse for mask
    t_c2 = data[mask0].T
    t_c3, mask1 = data_timecourse(data, [], mask0)
    # assert same for masks
    assert np.allclose(t_c2, t_c3, rtol=1e-4)
    assert np.allclose(mask0, mask1, rtol=1e-4)
    # get timecourse for thresholded voxels
    mean = data.mean(axis=-1)
    thresh = threshold_otsu(mean)
    mask0 = mean > thresh
    t_c4 = data[mask0].T
    t_c5, mask1 = data_timecourse(data, [], [])
    # assert same for all voxels
    assert np.allclose(t_c4, t_c5, rtol=1e-4)
    assert np.allclose(mask0, mask1, rtol=1e-4)

def test_create_design_matrix():
    # create random number trs, tr, and outliers
    n_tr = np.random.randint(1, 100)
    tr = np.random.randint(1, 4) + np.random.rand(1)
    outliers = np.sort(np.random.randint(n_tr, size=(10)))
    n_outliers = len(outliers)
    # create random timecourse of 0s and 1s with random n of columns
    col_n = np.random.randint(1, 10)
    t_c = np.random.randint(2,size=(n_tr, col_n))
    # create design matrix manually
    des0 = np.ones((n_tr, t_c.shape[1]))
    hrf_at_trs = spm_hrf(np.arange(0, 30, tr))
    n_to_remove = len(hrf_at_trs) - 1
    for i in range(t_c.shape[1]):
        convolved = np.convolve(t_c[:,i], hrf_at_trs)
        des0[:,i] = convolved[:-n_to_remove]
    # create R matrix
    R = np.zeros((n_tr, n_outliers))
    for i in range(n_outliers):
        R[outliers[i],i] = 1
    # get polynomial drift
    times = np.arange(0, tr*n_tr, tr)
    D = poly_drift(times)
    # stack des0, R, D
    des0 = np.column_stack([des0, R, D])
    # create design matrix from module
    des1 = create_design_matrix(t_c, tr, n_tr, outliers)
    # assert same
    assert np.allclose(des0, des1, rtol=1e-4)

def test_event_timecourse():
    # set test_events file name, condition_list, tr, and n_tr
    events_file = pjoin(MY_DIR, 'test_events.tsv')
    condition_list = ['hit', 'false_alarm', 'omiss', 'cor_rejec']
    tr = 2
    n_tr = 267
    # get test events and initialize timecourse
    task = np.genfromtxt(events_file, dtype='str')[1:, :]
    task[:, :2] = task[:, :2].astype(np.float) / tr
    tc0 = np.zeros((n_tr, len(condition_list)))
    # for each onset, set onset:onset+duration to 1
    for onset, duration, condition in task:
        onset = int(round(onset.astype(np.float)))
        duration = int(round(duration.astype(np.float)))
        tc0[onset:onset + duration, condition_list.index(condition)] = 1
    # get timecourse from function
    tc1 = event_timecourse(events_file, condition_list, tr, n_tr)
    # assert same timecourse
    assert np.allclose(tc0, tc1, rtol=1e-4)

def test_poly_drift():
    # load test_poly_drift.txt
    D0 = np.loadtxt(pjoin(MY_DIR,'test_poly_drift.txt'))
    # create times
    times = np.arange(0, 2*267, 2)
    # get matrix using poly_drift
    D1 = poly_drift(times, 3)
    # assert same
    assert np.allclose(D0, D1, rtol=1e-4)

def test_beta_res_calc():
    # get random data
    X0 = np.random.randn(100)
    Y0 = np.random.randn(100)
    # get beta values from scipy stats
    B0 = np.ones(2)
    B0[0], B0[1], _, _, _ = sst.linregress(X0, Y0)
    # get beta values and residuals using beta_res_calc
    X1 = np.ones((100,2))
    X1[:,0] = X0
    Y1 = Y0
    B1, e1 = beta_res_calc(Y1, X1)
    # calculate residuals
    e0 = Y1 - X1.dot(B0)
    # assert same betas and residuals
    assert np.allclose(B0, B1, rtol=1e-4)
    assert np.allclose(e0, e1, rtol=1e-4)

def test_compute_tstats():
    # create vol shape
    vol_shape = [64,64,30]
    mask = np.ones(vol_shape).astype(bool)
    # create random data and design matrix
    n = 60
    Y = np.random.randint(1, 100, size=(n, np.prod(vol_shape)))
    X = np.ones((n, 2))
    X[:int(n/2), 0] = 0
    X[int(n/2):, 1] = 0
    # calculate beta and residuals
    B = npl.pinv(X).dot(Y)
    res = Y - X.dot(B)
    # create contrast vector
    C = np.array([1, -1])
    # calculate df_error and sigma_2
    df_error = n - npl.matrix_rank(X)
    sigma_2 = np.sum(res ** 2, axis=0) / df_error
    # calculate c_b_cov
    c_b_cov = C.T.dot(npl.pinv(X.T.dot(X))).dot(C)
    # calculate t and p values
    t = C.T.dot(B) / np.sqrt(sigma_2 * c_b_cov)
    t_dist = sst.t(df_error)
    p = (1 - t_dist.cdf(np.abs(t))) * 2
    # reshape t and p values to tmap0 and pmap0
    tmap0 = np.zeros(vol_shape)
    pmap0 = np.zeros(vol_shape)
    tmap0[mask] = t
    pmap0[mask] = p
    # get tmap and pmap from compute_tstats
    tmap1, pmap1 = compute_tstats(C, X, B, res, mask)
    # assert same tmaps and pmaps
    assert np.allclose(tmap0, tmap1, rtol=1e-4)
    assert np.allclose(pmap0, pmap1, rtol=1e-4)
