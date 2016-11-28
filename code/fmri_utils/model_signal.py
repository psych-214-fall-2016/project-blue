# imports
import numpy as np
import numpy.linalg as npl
import nibabel as nib
from scipy.stats import gamma

def data_timecourse(data_fname, coord, mask=[], d=4):
    """ return data time course for coordinate or all voxels
    Parameters
    ----------
    data_fname : str, a data filename to return timecourse
    coord : array of coordinates or empty array to return all timecourses
    mask : array of voxels to include in analysis [default is all voxels]
    d : number of dummy trs to remove [optional; default = 4]

    Returns
    -------
    timecourse : array of timecourse for single voxel or all voxels with size of
    (n_trs, n_voxels)
     """
    # ensure coord is array
    coord = np.array(coord)
    # load data
    img = nib.load(data_fname)
    data = img.get_data()
    # get n_vols and vol_shape
    n_vols = data.shape[-1] - d
    vol_shape = data.shape[:-1]
    # remove dummy frames
    data = data[..., d:]
    # return time course for mask
    if len(mask) > 0:
        return data[mask].T
    # return time course for voxel
    elif len(coord) > 0:
        return data[coord[0],coord[1],coord[2]]
    else: # return all voxels
        return np.reshape(data, (np.prod(vol_shape), n_vols)).T

def create_design_matrix(time_course, tr, n_tr, d=4):
    """ return design matrix using event file
    Parameters
    ----------
    time_course : event time_course returned from event_timecourse function
    tr : number, repetition time in seconds
    n_tr : number, number of TRs
    d : number of dummy trs to remove [optional; default = 4]

    Returns
    -------
    X : array, design matrix (including column of 1s last)
    """
    # create design matrix
    X = np.ones((n_tr - d, time_course.shape[1] + 1))
    # create tr_times
    tr_times = np.arange(0, 30, tr)
    # get hrf_at_trs
    hrf_at_trs = hrf(tr_times)
    n_to_remove = len(hrf_at_trs) - 1
    # convolve for each column and input to design matrix
    for i in range(time_course.shape[1]):
        # convolve with hrf
        convolved = np.convolve(time_course[:,i], hrf_at_trs)
        convolved = convolved[d:-n_to_remove]
        # set convolved to X
        X[:,i] = convolved
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

def hrf(times):
    """ Return values for HRF at given times
    Parameters
    ----------
    times : number or array, timepoints to sample hrf

    Returns
    -------
    hrf : number or array, hrf values at given timepoints
    """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

def b_e_calc(Y, X):
    """ Return beta hat and residuals
    Parameters
    ----------
    Y : array, data to fit
    X : array, design matrix of predictors

    Returns
    -------
    B : array, beta hat values with size of (number of X columns, number of Y columns)
    e : array, residuals with size of (number of TRs, number of Y columns)
    """
    # calculate beta hat values
    B = npl.pinv(X).dot(Y)
    # calculate residuals
    e = Y - X.dot(B)
    return B, e

def create_contrast_img(B, C, vol_shape):
    """ Return Contrast image of B map
    Parameters
    ----------
    B : array, beta hat values
    C : array, contrast vector
    vol_shape : array, shape of volume

    Returns
    -------
    Bmap : contrast image of beta values
    """
    # ensure contrast is array
    C = np.array(C)
    # create Bmap
    Bmap = C.T.dot(B)
    # return reshaped image
    return np.reshape(Bmap, (vol_shape))
