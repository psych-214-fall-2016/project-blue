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

def get_contents(filename, var_list):
    """ Return variables within a file
    Parameters
    ----------
    filename : string
        filename to extract variables from
    var_list : list
        variables to extract from filename
    Returns
    -------
    outputs : list
        variables evaluated from filenames
    Example
    -------
    filename = 'task-visualimageryfalsememory_bold.json'
    var_list = ['RepetitionTime', 'TaskName', 'SliceTiming']
    outputs = get_contents(filename, var_list)
    outputs =
    [2.0,
     'Visual imagery false memory',
     [0.08,
      1.08,
      0.15,
      1.15,
      0.23,
      1.23,
      0.31,
      1.31,
      0.38,
      1.38,
      0.46,
      1.46,
      0.54,
      1.54,
      0.62,
      1.62,
      0.69,
      1.69,
      0.77,
      1.77,
      0.85,
      1.85,
      0.92,
      1.92,
      1.0,
      2.0]]
    """
    # open file
    fobj = open(filename, 'rt')
    contents = fobj.read()
    fobj.close()
    # create outputs
    outputs = list()
    # for each var in var_list, return value
    for var in var_list:
        m = re.search('"' + var + '": (.+),\n', contents)
        outputs.append(eval(m.groups()[0]))
    return outputs
