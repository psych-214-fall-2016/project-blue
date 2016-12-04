"""
The slice timing code will reslice a 4D image using the scipy
'InterpolatedUnivariateSpline' function.

INPUTS:
1) functional data that has gone through outlier detection
2) TR from scanning
3) other scanner info?

OUTPUTS:
1) new_data: Resliced, time corrected data
2) new_series: Corrected time series
"""

import numpy as np
import nibabel as nib

def slice_timing_corr(data, TR):
    #data = image.get_data()

    #configure matplotlib
    #plt.rcParams['image.cmap'] = 'gray'  # default gray colormap
    #plt.rcParams['image.interpolation'] = 'nearest'

    #get timing info
    vol_nos = np.arange(data.shape[-1])
    vol_onset_times = vol_nos * TR
    times_slice_0 = vol_onset_times
    n_z_slices = data.shape[2]
    time_for_single_slice = TR / n_z_slices
    time_course_slice_0 = data[:, :, 0, :]
    time_course_z = data[:, :, 1:n_z_slices, :]

    """
    Need to check that the below numbers are being generated correctly
    """
    acquisition_order = np.zeros(n_z_slices)
    acquisition_index = 0
    for i in range(0, n_z_slices, 2):
        acquisition_order[i] = acquisition_index
        acquisition_index += 1
    for i in range(1, n_z_slices, 2):
        acquisition_order[i] = acquisition_index
        acquisition_index += 1

    #- Divide acquisition_order by number of slices, multiply by TR
    time_offsets = acquisition_order / n_z_slices * TR

    #Copy old data into new array
    new_data = data.copy()

    from scipy.interpolate import InterpolatedUnivariateSpline

    #For each z coordinate (slice index):
    # Make `slice_z_times` vector for this slice
    ## For each x coordinate:
    ### For each y coordinate:
    #### extract the time series at this x, y, z coordinate;
    ##### make a linear interpolator object with the `slice_z_times` and
    #     the extracted time series;
    #### resample this interpolator at the slice 0 times;
    #### put this new resampled time series into the new data at the
    #    same position
    for z in range(data.shape[2]):
        slice_z_times = times_slice_0 + time_offsets[z]
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                time_series = data[x, y, z, :]
                interp = InterpolatedUnivariateSpline(slice_z_times, time_series, k=1)
                new_series = interp(times_slice_0)
                new_data[x, y, z, :] = new_series

    return new_data, new_series
