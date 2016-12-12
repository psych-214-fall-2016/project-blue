""" Wrapper for running code
Calls individual modules for analysis.
inputs:
data_folder = subject data location
"""

from __future__ import print_function, division
import os
import re
import sys
import numpy as np
import nibabel as nib
import numpy.linalg as npl
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from scipy.optimize import fmin_powell

import fmri_utils
import fmri_utils.slice_timing_corr
import fmri_utils.dir_utils

def run_analysis(data_folder):
    # The following parameters may be stored in a 'study_parameters' module,
    # but for now is called here at the top.
    # TR = 2, found in 'task-visualimageryfalsememory_bold.json'
    # There's some new code that I haven't had time to figure out that may read .json
    # files directly.
    TR = 2

    # 'event_file_index' initializes and advances the corresponding event file to the
    # current data file being analyzed
    event_file_index = 0
    # 'event_conditions' lists the event types found in each run's event file to
    # use in function model_signal.event_timecourse()
    event_conditions = ['hit', 'omiss', 'false_alarm', 'cor_rejec']

    # Each subject has 2 runs: the first is the encoding run, and the second is the
    # retrieval run.
    run01_files = fmri_utils.dir_utils.search_directory(data_folder, '.*run-01_bold.nii.gz')
    run02_files = fmri_utils.dir_utils.search_directory(data_folder, '.*run-02_bold.nii.gz')
    all_data_files = fmri_utils.dir_utils.search_directory(data_folder, '.*bold.nii.gz')

    # Each subject also has an events file associated with each run, which
    # contains the onsets/durations for the task and the event_type
    event01_files = fmri_utils.dir_utils.search_directory(data_folder, '.*run-01_events.tsv')
    event02_files = fmri_utils.dir_utils.search_directory(data_folder, '.*run-02_events.tsv')
    all_event_files = fmri_utils.dir_utils.search_directory(data_folder, '.*events.tsv')

    for data_file in all_data_files:

        # Load data
        img = nib.load(data_file)
        data = img.get_data()

        # Retrieve number of TRs in data
        number_TRs = data.shape[-1]

        # Validate data hashes

        # Find outliers as determined in detectors.py
        vol_mean, mean_outlier_arr = fmri_utils.detectors.mean_detector(data)
        vol_std, std_outliers_arr = fmri_utils.detectors.std_detector(data)

        # Add outliers found from IQR means and those from IQR std devs
        outliers_arr = mean_outlier_arr + std_outliers_arr

        # Make a dataset with removed outliers, if needed later, here.

        # Begin slice-timing correction found in slice_timing_corr.py
        time_corrected_data, time_corrected_series = fmri_utils.slice_timing_corr.slice_timing_corr(data, TR)

        # Apply motion correction from motion_correction.py
        # Use middle volume as the reference image
        reference_vol = int(time_corrected_data.shape[-1]/2)
        corrected_data = nib.Nifti1Image(time_corrected_data, img.affine)
        motion_corrected_data, motion_correction_params = fmri_utils.motion_correction.optimize_params(corrected_data, corrected_data, reference_vol)

        # Find array of timecourse for voxels; found in model_signal.py
        data_timecourse = fmri_utils.model_signal.data_timecourse(motion_corrected_data, [])

        # Make event timecourse
        event_timecourse = fmri_utils.model_signal.event_timecourse(all_event_files[event_file_index], event_conditions, TR, number_TRs)
        event_file_index = event_file_index + 1

        # Create design matrix from timecourse
        design_matrix = fmri_utils.model_signal.create_design_matrix(time_course, tr, n_tr, outliers=range(0,4))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Please enter the directory of the openfmri data')
        quit()
    data_folder = sys.argv[1]
    run_analysis(data_folder)
