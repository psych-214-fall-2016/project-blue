""" Test script for dir_utils module

Test with ``py.test test_dir_utils.py``.
"""

import os
import re
import hashlib

MY_DIR = os.path.dirname(__file__)
from fmri_utils import search_directory, get_contents, file_hash, validate_data

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

def test_file_hash():
    # set filename to test_dir_utils
    filename = os.path.join(MY_DIR, 'test_dir_utils.py')
    # get contents from filename
    fobj = open(filename, 'rb')
    contents = fobj.read()
    fobj.close()
    # get hash of file
    hash0 = hashlib.sha1(contents).hexdigest()
    # get hash with file_hash
    hash1 = file_hash(filename)
    # assert same
    assert hash0 == hash1

def test_validate_hash():
    # get data_hashes.txt
    filename = os.path.join(MY_DIR, 'data_hashes.txt')
    # get hashes and files
    fobj = open(filename, 'rt')
    lines = fobj.readlines()
    fobj.close()
    split_lines = [line.split() for line in lines]
    # check hashes are same as file hashes (except last line)
    for line in split_lines[:-1]:
        fhash = file_hash(MY_DIR + '/' + line[1])
        assert fhash == line[0]
    # run validate_data
    validate_data(MY_DIR)
