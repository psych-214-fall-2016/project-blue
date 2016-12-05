""" Functions for motion correcton.
"""
import sys
import numpy as np
import numpy.linalg as npl
import nibabel as nib
from scipy.ndimage import affine_transform
from scipy.optimize import fmin_powell

def create_rotmat(rotations):
    """ create rotation matrix from vector rotations

    Parameters
    ----------
    rotations : 1D array
        rotation vectors (in radians)

    Returns
    -------
    rot_mat : 2D array
        4 x 4 rotation matrix concatenated from x, y, z rotations
    """
    # set cos and sin of theta
    cos_t = [np.cos(x) for x in rotations]
    sin_t = [np.sin(x) for x in rotations]
    # x matrix
    rot_x = np.array([[1, 0, 0],
                      [0, cos_t[0], -sin_t[0]],
                      [0, sin_t[0], cos_t[0]]])
    # y matrix
    rot_y = np.array([[cos_t[1], 0, sin_t[1]],
                      [0, 1, 0],
                      [-sin_t[1], 0, cos_t[1]]])
    # z matrix
    rot_z = np.array([[cos_t[2], -sin_t[2], 0],
                      [sin_t[2], cos_t[2], 0],
                      [0, 0, 1]])
    # return concatenated rotations
    return rot_z.dot(rot_y).dot(rot_x)

def transformation_matrix(params):
    """ transformation_matrix returns a matrix defining a transformation.
    The transformation can be a translation, rotation, zooms.
    transformation given a vector of parameters (params).

    Parameters
    ----------
    params: 1D array
        vector of parameters
        params[0]  - x translation
        params[1]  - y translation
        params[2]  - z translation
        params[3]  - x rotation about - {pitch} (radians)
        params[4]  - y rotation about - {roll}  (radians)
        params[5]  - z rotation about - {yaw}   (radians)
        params[6]  - x zooms
        params[7]  - y zooms
        params[8]  - z zooms

    Returns
    -------
    affine: 2D array
        4 x 4 transformation matrix
    """
    # get translations
    vec = params[0:3]
    # get rotations
    rot = params[3:6]
    # get zooms
    zooms = params[6:9]
    # create zoom matrix
    zoom_mat = np.diag(zooms)
    # create rotation matrix
    rot_mat = create_rotmat(rot)
    # concat zoom and rot matrices
    M = zoom_mat.dot(rot_mat)
    # combine into affine
    return nib.affines.from_matvec(M, vec)

def apply_transform(vol, ref, params):
    """ Apply transformation in params to vol with output_shape ref

    Parameters
    ----------
    vol : 3D array
        volume to which to apply transformation
    ref : 3D array
        volume to use as output_shape during affine_transform
    params : 1D array
        vectors of translation, rotation, and zoom to apply to vol

    Returns
    -------
    resampled : 3D array
        resampled vol after applying transforamtion in params
    """
    # create affine from params
    P = transformation_matrix(params)
    # get template affine
    ref_vox2mm = ref.affine
    # get inverse subject affine
    mm2vol_vox = npl.inv(vol.affine)
    # concate affines
    Q = mm2vol_vox.dot(P).dot(ref_vox2mm)
    # get mat and vec from Q
    mat, vec = nib.affines.to_matvec(Q)
    # apply affine transform
    return affine_transform(vol, mat, vec, order=1, output_shape=ref.shape)

def cost_function(params, vol, ref):
    """ cost function for each affine transformation.

    Parameters
    ----------
    params : 1D array
        parameters for translation, rotation, and zooms
    vol : 3D array
        volume to move to align with ref
    ref : 3D array
        volume to which vol is aligned
    Returns
    -------
    correl : number
        negative correlation between the two volumes (which is
        lower when alignment is better)
    """
    # apply affine transformation on vol
    vol_resampled = apply_transform(vol, ref, params)
    # return negative correlation
    return -np.corrcoef(vol_resampled.ravel(), ref.ravel())[0, 1]

def my_callback(params):
    """ Callback for fmin_powell to display current params
    """
    print("Trying parameters " + str(params))

def optimize_params(data_img, ref_img, ref_idx=0):
    """ Optimize alignment params for each volume in data

    Parameters
    ----------
    data_img : nibabel image
        data with volumes to align with ref
    ref_img : nibabel image
        volume to be aligned with data volumes
    ref_idx : int
        index of ref_img data to use as reference volume

    Returns
    -------
    mc_data : 4D array
        motion corrected data following alignment with ref
    mc_params : 2D array
        motion params used to align each volume in data with ref
    """
    # get data from data_img and ref_img
    data = data_img.get_data()
    ref = ref_img.get_data()
    if len(data.shape) < 4:
        data = np.reshape(data, data.shape + (1,))
    if len(ref.shape) == 4:
        ref = ref[...,ref_idx]
    # set affines to data and ref
    data.affine = data_img.affine
    ref.affine = ref_img.affine
    # init mc_data and mc_params
    mc_data = np.zeros(ref.shape[:3] + (data.shape[-1],))
    mc_params = np.zeros((data.shape[-1], 9))
    # for each volume in data
    for i in range(data.shape[-1]):
        # set vol data [...,i] and affine
        vol = data[...,i]
        vol.affine = data.affine
        # run fmin_powell with data[..., i] and ref
        mc_params[i,:] = fmin_powell(cost_function, [0, 0, 0, 0, 0, 0, 1, 1, 1],
            args=(vol, ref), callback=my_callback)
        # apply transform
        mc_data[...,i] = apply_transform(vol, ref, mc_params[i])
    return mc_data, mc_params
