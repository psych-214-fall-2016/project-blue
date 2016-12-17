""" Model Signal module for analyzing fMRI data
"""

# imports
import numpy as np
import numpy.linalg as npl
import nibabel as nib
from fmri_utils import spm_hrf
import scipy.stats as sst
from scipy.stats import gamma

def data_timecourse(data, coord, mask=[]):
    """ return data time course for coordinate or all voxels
    Parameters
    ----------
    data : array of data to return as timecourse
    coord : array of coordinates or empty array to return all timecourses
    mask : array of voxels to include in analysis [default is all voxels]

    Returns
    -------
    timecourse : array of timecourse for single voxel or all voxels with size of
    (n_trs, n_voxels)
     """
    # ensure coord is array
    coord = np.array(coord)
    # get n_vols and vol_shape
    n_vols = data.shape[-1]
    vol_shape = data.shape[:-1]
    # return time course for mask
    if len(mask) > 0:
        return data[mask].T
    # return time course for voxel
    elif len(coord) > 0:
        return data[coord[0],coord[1],coord[2]]
    else: # return all voxels
        return np.reshape(data, (np.prod(vol_shape), n_vols)).T

def create_design_matrix(time_course, tr, n_tr, outliers=range(0,4)):
    """ return design matrix using event file
    Parameters
    ----------
    time_course : event time_course returned from event_timecourse function
    tr : number, repetition time in seconds
    n_tr : number, number of TRs
    outliers: indices of trs to remove [optional; default = range(0,4)]

    Returns
    -------
    X : array, design matrix with convolved time_course columns followed by
        outlier regressor columns, and a ones column
    """
    # get n_outliers
    n_outliers = len(outliers)
    # create design matrix
    X = np.ones((n_tr, time_course.shape[1]))
    # create tr_times
    tr_times = np.arange(0, 30, tr)
    # get hrf_at_trs
    hrf_at_trs = spm_hrf(tr_times)
    n_to_remove = len(hrf_at_trs) - 1
    # convolve for each column and input to design matrix
    for i in range(time_course.shape[1]):
        # convolve with hrf
        convolved = np.convolve(time_course[:,i], hrf_at_trs)
        X[:,i] = convolved[:-n_to_remove]
    # add outlier columns
    R = np.zeros((n_tr, n_outliers))
    for i in range(n_outliers):
        R[outliers[i],i] = 1
    # concatenate X, R, and ones column
    X = np.column_stack([X, R, np.ones((n_tr,))])
    return X

def event_timecourse(task_fname, condition_list, tr, n_tr):
    """ return time course from event files
    Parameters
    ----------
    task_fname : str, event filename containing onsets/durations for the task
    condition_list : list, condition names within task_fname column 3, indicating
    the condition for each corresponding event
    tr : number, repetition time in seconds
    n_tr : number, number of TRs

    Returns
    -------
    time_course : numpy array with columns in order of conditions in condition_list
    and rows corresponding to events in the fmri run.
    """
    # load task file
    task = np.genfromtxt(task_fname, dtype='str')
    # remove first row
    task = task[1:, :]
    # convert to trs
    task[:, :2] = task[:, :2].astype(np.float) / tr
    # initialize time_course
    time_course = np.zeros((n_tr, len(condition_list)))
    # for each onset, round onset and duration and set time_course
    for onset, duration, condition in task:
        onset = int(round(onset.astype(np.float)))
        duration = int(round(duration.astype(np.float)))
        # set time course for row events and column condition to 1
        time_course[onset:onset + duration, condition_list.index(condition)] = 1
    return time_course

def beta_res_calc(Y, X):
    """ Return beta hat and residuals
    Parameters
    ----------
    Y : array, data to fit
    X : array, design matrix of predictors

    Returns
    -------
    B : array, beta hat values with size of (number of X columns, number of Y columns)
    res : array, residuals with size of (number of TRs, number of Y columns)
    """
    # calculate beta hat values
    B = npl.pinv(X).dot(Y)
    # calculate residuals
    res = Y - X.dot(B)
    return B, res

def compute_tstats(C, X, B, res, vol_shape):
    """ Computes t statistics and p-values and returns 3D t_map and p_map
    Parameters
    ----------
    C : array
        contrast vector to test beta values
    B : array
        beta hat values calculated from beta_res_calc
    X : array
        design matrix used in GLM
    res : array
        residuals calculated from beta_res_calc
    vol_shape : numpy array
        shape of output volume to reshape t and p values

    Returns
    -------
    t_map : array shape (vol_shape)
        t statistics for each voxel
    p_map : array shape (vol_shape)
        two-tailed probability value for each t statistic
    """
    # calculate df_error
    n = X.shape[0]
    x_rank = npl.matrix_rank(X)
    df_error = n - x_rank
    # calculate sigma_2 and c_b_cov
    sigma_2 = np.sum(res ** 2, axis=0) / df_error
    c_b_cov = C.T.dot(npl.pinv(X.T.dot(X))).dot(C)
    # calculate t statistics
    t = C.T.dot(B) / np.sqrt(sigma_2 * c_b_cov)
    # calculate p values
    t_dist = sst.t(df_error)
    p = (1 - t_dist.cdf(np.abs(t))) * 2
    # reshape t and p values to volume shape
    return np.reshape(t, vol_shape), np.reshape(p, vol_shape)
