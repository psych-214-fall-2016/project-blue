# project-blue

Psych 214 Fall 2016 Final Project

[![Build Status](https://travis-ci.org/psych-214-fall-2016/project-blue.svg?branch=master)](https://travis-ci.org/psych-214-fall-2016/project-blue)

This repository stores our fMRI methods course project investigating false visual memory. We took a dataset from openfMRI collected by Stephan-Otto and colleagues (https://openfmri.org/dataset/ds000203/). In the experiment, 26 healthy subjects were shown to images and words (run01, encoding), and later asked to recall whether they had seen an image or a word (run02, retrieval). 
Our analysis consists of four steps: 
1) preprocessing of the data
2) contrasting retrieval (run02) [Hits + Correct Rejections] vs [Misses + False Alarms], to get an ROI/mask for each subject, 
3) within the mask, contrasting encoding (run01) subsequent [False Alarms] vs [Correct Rejections] to generate a t-value for each subject, 
4) regressing t-values with subjective ratings of visual imagery. 
Through this analysis, we aim to determine whether differences in brain activation during inaccurate memory retrieval are associated with individual differences in visual imagery.

Thanks to Matthew Brett, JB Poline, and Arielle Tambini for their guidance with this project

## Install

To install the necessary code:

    # Install required packages
    pip3 install --user -r requirements.txt
    # Put code/fmri_utils onto Python path using setup.py
    pip3 install --user --editable ./code

## Test

Install pytest:

    pip3 install --user pytest

Run the tests:

    py.test code

## Download data
Overview:

All data and the experiment overview can be found here: https://openfmri.org/dataset/ds000203/

Only download data from latest revision, in this case, Revision: 1.0.1 Date Set: Oct. 17, 2016.

Subject data is found in link "Data for All Subjects (1-26)" with filename 'ds000203_R1.0.1_data.zip'.

The directory you choose to save the data in will be used to run the wrapper code 'wrapper.py', which executes each step of the analysis sequence. Make note of this location on your local drive.

Download data:

cd data/
curl -LO http://openfmri.s3.amazonaws.com/tarballs/ds000203_R1.0.1_metadata.zip
unzip ds000203_R1.0.1_metadata.zip
rm ds000203_R1.0.1_metadata.zip
curl -LO http://openfmri.s3.amazonaws.com/tarballs/ds000203_R1.0.1_data.zip
unzip ds000203_R1.0.1_data.zip
rm ds000203_R1.0.1_data.zip
cd ..

## Validate data

scripts/validate_data.py data

## Run analysis

To run the analysis, first download the project-blue repository and add its download location to your python path.

Run wrapper.py:

    python scripts/wrapper.py data data 1
    
## Contributors

Melissa Newton
Peiwu Qin
Justin Theiss
Joe Winer
