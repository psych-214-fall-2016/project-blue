"""
The slice timing code reslices a 4D image using the scipy
'InterpolatedUnivariateSpline' function.

This function uses the linear interpolation formula:
y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)

INPUTS:
1) functional data that has gone through outlier detection
2) TR from scanning

OUTPUT:
1) new_data: Resliced, time corrected data
"""

import numpy as np
import nibabel as nib

def slice_timing_corr(data, TR):

    #import timing info from openfmri json file
    outputs = get_contents('task-visualimageryfalsememory_bold.json', ['SliceTiming'])
    time_offsets = outputs[0]
    #time_offsets = [0.08, 1.08, 0.15, 1.15, 0.23, 1.23, 0.31, 1.31, 0.38, 1.38, 0.46, 1.46, 0.54, 1.54, 0.62, 1.62, 0.69, 1.69, 0.77, 1.77, 0.85, 1.85, 0.92, 1.92, 1.00, 2.00]

    #get timing info
    vol_nos = np.arange(data.shape[-1])
    vol_onset_times = vol_nos * TR
    n_z_slices = data.shape[2]
    time_for_single_slice = TR / n_z_slices
    time_course_z = data[:, :, 1:n_z_slices, :]
    times_slice_0 = np.arange(data.shape[-1]) * TR

    #Copy old data into new array
    new_data = data.copy()

    from scipy.interpolate import InterpolatedUnivariateSpline

    #For each slice (z)
    for z in range(data.shape[2]):
        #Calculate the time series of acquisition
        slice_z_times = times_slice_0 + time_offsets[z]
        #For each x coordinate
        for x in range(data.shape[0]):
            #For each y coordinate
            for y in range(data.shape[1]):
                #Get the time series at this x y z coordinate
                time_series = np.array(data[x, y, z, :])
                ##### make a linear interpolator object with the `slice_z_times` and
                #     the extracted time series;
                interp = InterpolatedUnivariateSpline(slice_z_times, time_series, k=1)
                #### resample this interpolator at the slice 0 times;
                new_series = interp(times_slice_0)
                #### put this new resampled time series into the new data at the
                #    same position
                new_data[x, y, z, :] = new_series

    return new_data
