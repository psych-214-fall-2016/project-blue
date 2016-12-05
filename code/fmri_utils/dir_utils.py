""" Utilities for working with directories
"""

# import os and regex
import os
import re

def search_directory(start_dir, expr):
    """ Returns fullpath filenames with expr within start_dir
    Parameters
    ----------
    start_dir : string
        starting directory to begin searching for files
    expr : string
        regular expression used to search for files
    Returns
    -------
    file_paths : list
        list of fullpath filenames with expression expr within start_dir
    Example
    -------
    Return all python files that begin with 'test' starting within
    current directory ('/Users/project-blue/code/fmri_utils')
    file_paths = search_directory('.', 'test.*\.py')
    ['/Users/project-blue/code/fmri_utils/tests/test_detectors.py',
    '/Users/project-blue/code/fmri_utils/tests/test_dir_utils.py',
    '/Users/project-blue/code/fmri_utils/tests/test_model_signal.py',
    '/Users/project-blue/code/fmri_utils/tests/test_motion_correction.py',
    '/Users/project-blue/code/fmri_utils/tests/test_slice_timing_corr.py',
    '/Users/project-blue/code/fmri_utils/tests/test_spm_funcs.py']
    """
    # initialize file_paths
    file_paths = list()
    # get fullpath of start_dir
    start_dir = os.path.abspath(start_dir)
    # get new paths within start_dir
    new_paths = os.listdir(start_dir)
    new_paths = [os.path.join(start_dir, i) for i in new_paths]
    # for each new path, check if isdir
    for d in new_paths:
        # if isdir, run search_directory
        if os.path.isdir(d):
            tmp_paths = search_directory(d, expr)
            [file_paths.append(i) for i in tmp_paths]
        # if expression found, set to file_paths
        elif re.search(expr, d) != None:
            file_paths.append(d)
    return file_paths
