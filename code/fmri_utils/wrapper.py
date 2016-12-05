""" Wrapper for running code
Calls individual modules for analysis.
inputs:
data_folder = subject data location
"""

from __future__ import print_function, division
import os
import re
import numpy as np
import nibabel as nib
import numpy.linalg as npl
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from scipy.optimize import fmin_powell

import fmri_utils
import fmri_utils.slice_timing_corr

# In order to run this wrapper, you must provide the directory of the local .nii data.
# It will take in the location with something like: data_folder = args[0]
# For the purposes of writing and testing, I have hard coded data_folder to
# the location on my personal computer on the following line.
data_folder = '/Users/kayserlab/Documents/Psych214/data'
subject_folders = os.listdir(data_folder)

subject_folders = [s for s in subject_folders if s.startswith("sub")]

for subject_folder in subject_folders:
    subject_dir = os.path.join(data_folder, subject_folder, "func")
    subject_files = os.listdir(subject_dir)
    nii_files = [s for s in subject_files if s.endswith(".nii.gz")]

    # Next line will use only the first run from subject for model_signal.py as instructed in outline_for_wrapper
    # nii_files = [s for s in nii_files if re.search("run-01", s) is not None]
    for nii_file in nii_files:
        nii_filepath = os.path.join(subject_dir, nii_file)

        img = nib.load(nii_filepath)
        raw_data = img.get_data()

        # Find array of timecourse for voxels; found in model_signal.py
        timecourse_data = fmri_utils.model_signal.data_timecourse(nii_filepath, [])

        # Find outliers as determined in detectors.py
        vol_mean, mean_outliers = fmri_utils.detectors.mean_detector(raw_data)
        vol_std, std_outliers = fmri_utils.detectors.std_detector(raw_data)

        # Add outliers found from IQR means and those from IQR std devs
        all_outliers = mean_outliers + std_outliers
        all_inliers = [x for x in range(raw_data.shape[-1]) if x not in all_outliers]

        # Begin using data without the volumes that were identified as outliers
        data = raw_data[...,all_inliers]

        # Begin slice-timing correction found in slice_timing_corr.py
        # TR = 2, found in 'task-visualimageryfalsememory_bold.json'
        TR = 2
        time_correct_data, time_correct_series = fmri_utils.slice_timing_corr.slice_timing_corr(data, TR)

        # Apply motion correction from motion_correction.py
        # Clarifications need to be made for motion_correction to run--
        # For example, cost_function(matvec) calls nib.affines.to_matvec
        # but never uses the resultant outputs mat or vec in its return of
        # variable '-correl'. The rest of this wrapper code will use a dummy
        # variable "motion_correct_data" that assumes the output of motion_correction.py,
        # although this is currently not a real output.

        
