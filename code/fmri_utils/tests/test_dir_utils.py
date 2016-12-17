""" Test script for dir_utils module

Test with ``py.test test_dir_utils.py``.
"""

import os
import re
import numpy as np
import hashlib

MY_DIR = os.path.dirname(__file__)
from fmri_utils import search_directory, get_contents, dlm_read, file_hash, validate_data

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

def test_dlm_read():
    # check that Age is the same as below
    filename = os.path.join(MY_DIR, 'test_dlm_read.tsv')
    column = ['test2']
    outputs0 = [np.array([0,1,2,3,4,5,6,7,8,9]), np.array(['a','b','c','d',
    'e','f','g','h','i','j'])]
    # get outputs using dlm_read
    outputs1 = dlm_read(filename, '\t', column)[0]
    # assert same first test
    assert np.all(outputs0[1] == outputs1)
    # test all columns
    outputs1 = dlm_read(filename, '\t', [])
    # assert same all columns
    assert np.all(outputs0[0] == outputs1[0])
    assert np.all(outputs0[1] == outputs1[1])

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
    # get data_hashes.txt and test files
    filename = os.path.join(MY_DIR, 'data_hashes.txt')
    files = ['test_events.tsv','test_task.json']
    # get hashes for files
    hashes = list()
    for f in files:
        hashes.append(file_hash(MY_DIR + '/' + f))
    # write hashes and files to data_hashes
    fobj = open(filename, 'wt')
    for x in range(len(files)):
        fobj.write(hashes[x] + ' ' + files[x] + '\n')
    # include incorrect hash/file pair
    fobj.write('0 test_dir_utils.py')
    fobj.close()
    # test validate_data when not all correct
    validate_data(MY_DIR)
    fobj = open(filename, 'wt')
    for x in range(len(files)):
        fobj.write(hashes[x] + ' ' + files[x] + '\n')
    fobj.close()
    # test validate_data with all correct
    validate_data(MY_DIR)
