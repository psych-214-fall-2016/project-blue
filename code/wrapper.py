""" Wrapper for analyzing fMRI data
This wrapper estiamtes a GLM for each subject's second run (retrieval) to obtain
a p-value map which is thresholded at 0.01 as a mask for the second analysis
using the contrast of correct (hit/correct rejection) greater than incorrect
(false alarm/omission).
For the second analysis, the GLM for each subject's first run (encoding) is
estimated within the first analysis mask based on the contrast of false alarm
greater than correct rejection. At this point, the t values from the
masked voxels are averaged. Once a t value has been obtained for each subject,
Pearson correlations are calculated between the t values and the following
clinical scores: Launay-Slade Hallucination Scale, Visual imagery score, and
Peters Delusion Inventory scale.

The following modules are run when analyzing data:
dir_utils (for getting filenames and retrieving study/participant data)
detectors (for outlier detection)
slice_timing_corr (for slice timing correction)
motion_correction (for motion correction)
model_signal (for estimating GLM)

Parameters
----------
data_dir : string, directory containing subject folders
param_dir : string, directory containing task-visualimageryfalsememory_bold and
    participants.tsv files
verbose : int, optional input of 1 or 0 to indicate whether information will be
    displayed in the command window (default is 1, true)
Returns
-------
None
"""

from __future__ import print_function, division
import os
import sys
import nibabel as nib
import numpy as np
import scipy.stats as sst
import matplotlib.pyplot as plt
from fmri_utils import *

def run_analysis(data_dir, params_dir, verbose=True):
    # load params file, get TR and slice time information
    params_file = search_directory(params_dir, 'task-.*.json')[0]
    print_verbose('Loading parameters from ' + params_file, verbose)
    contents = get_contents(params_file,['RepetitionTime','SliceTiming'])
    tr = contents[0]
    slicetiming = contents[1]
    print_verbose('TR: ' + str(tr), verbose)
    print_verbose('SliceTiming: ', verbose)
    print_verbose(slicetiming, verbose)

    # get subject clinical data
    clinical_file = search_directory(params_dir, 'participants.tsv')[0]
    clinical_scales = ['LSHS_14','DELUproneness']
    clinical_data = dlm_read(clinical_file,'\t',clinical_scales)
    for x in range(len(clinical_scales)):
        print_verbose(clinical_scales[x] + ' scores:', verbose)
        print_verbose(clinical_data[x], verbose)

    # 'event_conditions' lists the event types found in each run's event file to
    # use in function model_signal.event_timecourse()
    event_conditions = ['hit', 'cor_rejec', 'false_alarm', 'omiss', 'miss']
    print_verbose('Conditions: ', verbose)
    print_verbose(event_conditions, verbose)

    # Each subject has 2 runs: the first is the encoding run, and the second is the
    # retrieval run.
    run01_files = search_directory(data_dir, '.*run-01_bold.nii.gz')
    run02_files = search_directory(data_dir, '.*run-02_bold.nii.gz')

    # Each subject also has an events file associated with each run, which
    # contains the onsets/durations for the task and the event_type
    event01_files = search_directory(data_dir, '.*run-01_events.tsv')
    event02_files = search_directory(data_dir, '.*run-02_events.tsv')
    # create empty t_values variable
    t_values = np.zeros(len(run01_files))

    # for each subject, analyze both fmri runs
    for i in range(len(run01_files)):
        # get data_files, event_files, contrasts
        data_files = [run02_files[i], run01_files[i]]
        event_files = [event02_files[i], event01_files[i]]
        contrasts = [np.array([1,1,-1,-1]), np.array([0,-1,1,0])]

        # for each data_file, load data and run analysis
        for x, data_file in enumerate(data_files):
            # Load data
            print_verbose('Loading ' + data_file, verbose)
            img = nib.load(data_file)
            data = img.get_data()

            # Retrieve number of TRs in data
            n_tr = data.shape[-1]
            print_verbose('Number of TRs: ' + str(n_tr), verbose)

            # Find outliers as determined in detectors.py
            vol_mean, mean_outliers = mean_detector(data)
            vol_std, std_outliers = std_detector(data)
            print_verbose('Mean outliers: ', verbose)
            print_verbose(mean_outliers, verbose)
            print_verbose('Standard devation outliers: ', verbose)
            print_verbose(std_outliers, verbose)

            # Add outliers found from IQR means and those from IQR std devs
            outliers = np.unique(mean_outliers + std_outliers)
            print_verbose('Overall outliers:', verbose)
            print_verbose(outliers, verbose)

            # Begin slice-timing correction found in slice_timing_corr.py
            print_verbose('Running slice time correction...', verbose)
            stc_data, stc_series = slice_timing_corr(data, tr, slicetiming)

            # Apply motion correction from motion_correction.py
            # Use middle volume as the reference image
            print_verbose('Running motion correction...', verbose)
            ref_vol = int(stc_data.shape[-1]/2)
            data_img = nib.Nifti1Image(stc_data, img.affine)
            mc_data, mc_params = optimize_params(data_img, data_img, ref_vol)
            print_verbose('Final parameters: ', verbose)
            print_verbose(mc_params, verbose)

            # set mask to empty if first run
            if x == 0:
                mask = []
            # get data timecourse
            Y, mask = data_timecourse(mc_data, [], mask)

            # get event timecourse
            event_tc = event_timecourse(event_files[x], event_conditions, tr, n_tr)

            # Create design matrix from timecourse
            X = create_design_matrix(event_tc, tr, n_tr, outliers)

            # calculate beta values and residuals
            print_verbose('Estimating beta values...', verbose)
            B, res = beta_res_calc(Y, X)

            # create contrast
            C = contrasts[x]
            print_verbose('Contrast: ', verbose)
            print_verbose(C, verbose)

            # return t_map and p_map
            print_verbose('Computing statistical maps...', verbose)
            print_verbose('Mask size: ' + str(mask.astype(float).sum()) + ' voxels', verbose)
            t_map, p_map = compute_tstats(C, X, B, res, mask)

            # create mask based on p_map
            if x == 0:
                pthr = 0.01
                print_verbose('Thresholding p-value map at: ' + str(pthr), verbose)
                mask = mask & (p_map <= pthr)
            else: # get mean t value
                t_values[i] = t_map[mask].max() # .mean()
                print_verbose('Mean T-value: ' + str(t_values[i]), verbose)
                if np.isnan(t_values[i]):
                    print_verbose('Error: this subject will not be included in analysis', verbose)

    # run correlations with t_values
    for i, scores in enumerate(clinical_data):
        # check for any nans
        idx = np.array([np.isnan(x) == False for x in t_values])
        idx = idx & np.array([np.isnan(x) == False for x in scores])
        print_verbose('Excluding subject(s) from analysis: ', verbose)
        print_verbose([i for i, x in enumerate(idx) if x == False], verbose)
        # correlate t values and clinical data
        r, p = sst.pearsonr(t_values[idx], scores[idx])
        print_verbose('Correlation with ' + clinical_scales[i] + ':', verbose)
        print_verbose('r = ' + str(r) + ' p = ' + str(p), verbose)

def print_verbose(msg, verbose):
    """ Print message if verbose is True
    Parameters
    ----------
    msg : string, message to print
    verbose : bool, True/False whether to print message
    Returns
    -------
    None
    """
    # if verbose, print
    if verbose:
        print(msg)

if __name__ == '__main__':
    # if not enough arguments, raise error
    if len(sys.argv) < 3:
        raise ValueError('Please enter the data directory and parameter file')

    # get data_dir and params_file from inputs
    data_dir = sys.argv[1]
    params_dir = sys.argv[2]
    if len(sys.argv) == 4:
        verbose = np.array(sys.argv[3]).astype(bool)
    else:
        verbose = True

    # run analysis
    run_analysis(data_dir, params_dir, verbose)
