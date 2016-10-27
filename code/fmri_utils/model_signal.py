# imports
import numpy as np
import numpy.linalg as npl
import nibabel as nib
from scipy.stats import gamma

def data_timecourse(data_fname, coord, mask=[], d=4):
    """ return data time course for coordinate or all voxels
    Inputs:
    -------
    data_fname : str, a data filename to return timecourse
    coord : array of coordinates or empty array to return all timecourses
    mask : array of voxels to include in analysis [default is all voxels]
    d : number of dummy trs to remove [optional; default = 4]
    Output:
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
    vol_shape = data.shape[:3]
    # if no mask, create
    if len(mask) == 0:
        mask = np.ones(data.shape[:3], dtype=bool)
    # remove first n dummies and constrict with mask
    data = data[mask, d:]
    data = np.reshape(data, vol_shape + (n_vols,))
    # return time course for voxel
    if len(coord) > 0:
        return data[coord[0],coord[1],coord[2]]
    else:
        return np.reshape(data, (np.prod(vol_shape), n_vols)).T

def create_design_matrix(task_fname, tr, n_tr, d=4):
    """ return design matrix using event file
    Inputs:
    -------
    task_fname : list or str of event filename(s) containing the onsets/durations
    for the task
    tr : number, repetition time in seconds
    n_tr : number, number of TRs
    d : number of dummy trs to remove [optional; default = 4]
    Outputs:
    --------
    X : array, design matrix (including column of 1s)
    """
    # ensure task_fname is list
    if type(task_fname) != list:
        task_fname = list([task_fname])
    # create design matrix
    X = np.ones((n_tr - d, len(task_fname) + 1))
    # create tr_times
    tr_times = np.arange(0, 30, tr)
    # get hrf_at_trs
    hrf_at_trs = hrf(tr_times)
    # for each file, get time_course and convolve with hrf
    for i, fname in enumerate(task_fname):
        # get event time course
        time_course = event_timecourse(fname, tr, n_tr)
        # convolve with hrf
        convolved = np.convolve(time_course, hrf_at_trs)
        n_to_remove = len(hrf_at_trs) - 1
        convolved = convolved[d:-n_to_remove]
        # set convolved to X
        X[:,i] = convolved
    return X

def event_timecourse(task_fname, tr, n_tr):
    """ return time course from event files
    Inputs:
    -------
    task_fname : str, event filename containing onsets/durations for the task
    tr : number, repetition time in seconds
    n_tr : number, number of TRs
    """
    # load task file
    task = np.loadtxt(task_fname)
    # convert to trs
    task[:, :2] = task[:, :2] / tr
    # initialize time_course
    time_course = np.zeros(n_tr)
    # for each onset, round onset/duration and set time_course
    for onset, duration, amplitude in task:
        onset = int(round(onset))
        duration = int(round(duration))
        time_course[onset:onset + duration] = amplitude
    return time_course

def hrf(times):
    """ Return values for HRF at given times
    Inputs:
    -------
    times : number or array, timepoints to sample hrf
    Outputs:
    --------
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
    Inputs:
    -------
    Y : array, data to fit
    X : array, design matrix of predictors
    Outputs:
    --------
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
    Inputs:
    -------
    B : array, beta hat values
    C : array, contrast vector
    vol_shape : array, shape of volume
    Outputs:
    --------
    Bmap : contrast image of beta values
    """
    # ensure contrast is array
    C = np.array(C)
    # create Bmap
    Bmap = C.T.dot(B)
    # return reshaped image
    return np.reshape(Bmap.T, (data_shape))
