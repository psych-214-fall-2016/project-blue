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

## Download data

All data and the experiment overview can be found here: https://openfmri.org/dataset/ds000203/

Only download data from latest revision, in this case, Revision: 1.0.1 Date Set: Oct. 17, 2016.

Subject data is found in link "Data for All Subjects (1-26)" with filename 'ds000203_R1.0.1_data.zip'.

The directory you choose to save the data in will be used to run the wrapper code 'wrapper.py', which executes each step of the analysis sequence. Make note of this location on your local drive.

## Run analysis

To run the analysis, first download the project-blue repository and add its download location to your python path. Open ipython (or python) and import the fmri_utils module as so:

    from fmri_utils.wrapper import run_analysis

Then call the run_analysis function as so:

    run_analysis('directory_of_openfmri_data_you_downloaded')
