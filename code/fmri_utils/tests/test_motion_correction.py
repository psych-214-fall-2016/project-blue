""" Test script for motion_correction module

Test with ``py.test test_motion_correction.py``.
"""

from os.path import dirname, join as pjoin

MY_DIR = dirname(__file__)
EXAMPLE_FILENAME = 'ds107_sub012_t1r2_small.nii'
EXAMPLE_FULLPATH = pjoin(MY_DIR, EXAMPLE_FILENAME)

import numpy as np
import numpy.linalg as npl
import nibabel as nib
from scipy.ndimage import affine_transform
from scipy.optimize import fmin_powell

# This import needs the code directory on the Python PATH
from fmri_utils import create_rotmat, transformation_matrix, \
     apply_transform, cost_function, optimize_params

# set random seed
np.random.seed(1990)

def test_create_rotmat():
    # create random rotations
    rotations = np.random.rand(3)
    # x matrix
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(rotations[0]), -np.sin(rotations[0])],
                      [0, np.sin(rotations[0]), np.cos(rotations[0])]])
    # y matrix
    rot_y = np.array([[np.cos(rotations[1]), 0, np.sin(rotations[1])],
                      [0, 1, 0],
                      [-np.sin(rotations[1]), 0, np.cos(rotations[1])]])
    # z matrix
    rot_z = np.array([[np.cos(rotations[2]), -np.sin(rotations[2]), 0],
                      [np.sin(rotations[2]), np.cos(rotations[2]), 0],
                      [0, 0, 1]])
    # concatenate rotation matrices
    rot_mat0 = rot_z.dot(rot_y).dot(rot_x)
    # get rotation matrix from create_rotmat
    rot_mat1 = create_rotmat(rotations)
    # assert same
    assert np.allclose(rot_mat0, rot_mat1, rtol=1e-4)

def test_transformation_matrix():
    # create random params
    params = np.random.rand(9)
    # split into vec, rot zooms
    vec = params[0:3]
    rot = params[3:6]
    zooms = params[6:]
    # make matrix
    zoom_mat = np.diag(zooms)
    rot_mat = create_rotmat(rot)
    M = zoom_mat.dot(rot_mat)
    # make affine matrix
    affine_mat0 = nib.affines.from_matvec(M, vec)
    # get matrix from transformation_matrix
    affine_mat1 = transformation_matrix(params)
    # assert same
    assert np.allclose(affine_mat0, affine_mat1, rtol=1e-4)

def test_apply_transform():
    # create random params
    params = np.random.rand(9)
    P = transformation_matrix(params)
    # get example volume
    test_img = nib.load(EXAMPLE_FULLPATH)
    test_data = test_img.get_data()
    vol = test_data[...,0]
    ref = test_data[...,1]
    vol_img = nib.Nifti1Image(vol, test_img.affine)
    ref_img = nib.Nifti1Image(ref, test_img.affine)
    # concatenate matrices then get mat, vec
    Q = npl.inv(vol_img.affine).dot(P).dot(ref_img.affine)
    mat, vec = nib.affines.to_matvec(Q)
    # apply affine to create resampled data
    resampled0 = affine_transform(vol, mat, vec, order=1, output_shape=ref.shape)
    # get resampled data from apply_transform
    resampled1 = apply_transform(vol_img, ref_img, params)
    # assert same
    assert np.allclose(resampled0, resampled1, rtol=1e-4)

def test_cost_function():
    # get test data
    test_img = nib.load(EXAMPLE_FULLPATH)
    test_data = test_img.get_data()
    vol = test_data[...,0]
    ref = test_data[...,1]
    vol_img = nib.Nifti1Image(vol, test_img.affine)
    ref_img = nib.Nifti1Image(ref, test_img.affine)
    # create random params
    params = np.random.rand(9)
    # apply transformation
    resampled = apply_transform(vol_img, ref_img, params)
    # get negative correlation
    correl0 = -np.corrcoef(resampled.ravel(), ref.ravel())[0, 1]
    # get negative correlation from cost_function
    correl1 = cost_function(params, vol_img, ref_img)
    # assert same
    assert np.allclose(correl0, correl1, rtol=1e-4)

def test_optimize_params():
    # load test data
    test_img = nib.load(EXAMPLE_FULLPATH)
    test_data = test_img.get_data()
    # create translation in random volume (after first)
    idx = np.random.randint(1, test_data.shape[-1])
    trans = [1, 1, 1]
    test_data[...,idx] = affine_transform(test_data[...,idx], np.eye(3), trans, order=1)
    vol = test_data[...,idx]
    ref = test_data[...,0]
    vol_img = nib.Nifti1Image(vol, test_img.affine)
    ref_img = nib.Nifti1Image(ref, test_img.affine)
    # get motion correction params
    mc_params0 = fmin_powell(cost_function, [0, 0, 0, 0, 0, 0, 1, 1, 1],
        args=(vol_img, ref_img))
    # apply mc_params
    mc_data0 = apply_transform(vol_img, ref_img, mc_params0)
    # get motion corrected data and params using optimize_params
    vol_img = nib.Nifti1Image(test_data, test_img.affine)
    ref_img = nib.Nifti1Image(np.reshape(ref, ref.shape + (1,)), test_img.affine)
    mc_data1, mc_params1 = optimize_params(vol_img, ref_img, 0)
    # assert same data and params
    assert np.allclose(mc_data0, mc_data1[...,idx], rtol=1e-4)
    assert np.allclose(mc_params0, mc_params1[idx,:], rtol=1e-4)
    # test single image
    vol_img = nib.Nifti1Image(vol, test_img.affine)
    ref_img = nib.Nifti1Image(ref, test_img.affine)
    mc_data2, mc_params2 = optimize_params(vol_img, ref_img)
    # assert same data and params
    assert np.allclose(mc_data0, mc_data2[...,0], rtol=1e-4)
    assert np.allclose(mc_params0, mc_params2, rtol=1e-4)
