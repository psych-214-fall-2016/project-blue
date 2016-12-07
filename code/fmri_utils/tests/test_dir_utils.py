""" Test script for dir_utils module

Test with ``py.test test_dir_utils.py``.
"""

import os
import re

MY_DIR = os.path.dirname(__file__)
from fmri_utils import search_directory, get_contents

def test_search_directory():
    # get fullpath for filename
    expr = __file__
    start_dir = os.path.abspath('.')
    file_paths0 = list([os.path.join(start_dir, expr)])
    # get fullpath for filename using search_directory
    file_paths1 = search_directory(start_dir, expr)
    # assert same
    assert file_paths0 == file_paths1

def test_get_contents():
    # check that TR, TaskName, and Manufacturer are same
    var_list = ['RepetitionTime', 'TaskName', 'Manufacturer']
    outputs0 = [2.0, 'Visual imagery false memory', 'General Electric']
    filename = os.path.join(MY_DIR, 'test_task.json')
    # get outptus from filename
    outputs1 = get_contents(filename, var_list)
    # assert same
    assert outputs0 == outputs1
