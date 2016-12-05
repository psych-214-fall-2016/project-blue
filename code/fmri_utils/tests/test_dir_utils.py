""" Test script for dir_utils module

Test with ``py.test test_dir_utils.py``.
"""

import os
import re

from fmri_utils import search_directory

def test_search_directory():
    # get fullpath for filename
    expr = __file__
    start_dir = os.path.abspath('.')
    file_paths0 = list([os.path.join(start_dir, expr)])
    # get fullpath for filename using search_directory
    file_paths1 = search_directory(start_dir, expr)
    # assert same
    assert file_paths0 == file_paths1
