# project-template

Fall 2016 final project.

This file is in [Markdown
format](http://daringfireball.net/projects/markdown), and should render nicely
on the Github front page for this repository.

## Install

To install the necessary code:

    # Install required packages
    pip3 install --user -r requirements.txt
    # Put code/fmri_utils onto Python path using setup.py
    pip3 install --user --editable ./code

To run tests:

* install `pytest` with ``pip3 install --user pytest``;
* run tests with:

    py.test fmri_utils

## Test

Install pytest:

    pip3 install --user pytest

Run the tests:

    py.test code
