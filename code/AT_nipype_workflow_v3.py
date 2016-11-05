"""Test workflow modified by CLG from test_workflow_at.py in 
/home/despo/arielle/simpace_testing/ 
"""

import os, sys, json
from glob import glob
from os.path import join as pjoin
from matplotlib import pyplot as plt
import nipype.interfaces.spm as spm
import nipype.interfaces.matlab as matlab
import nipype.interfaces.utility as utility
import nipype.pipeline.engine as pe
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.utility import Function
from nipype.interfaces import afni

matlab.MatlabCommand.set_default_paths('/usr/local/matlab-tools/spm/spm8')

# 1st arg: 01 (sub-01)
# 2nd argument: trajectory (e.g. A)
# 3rd argument: 01 (ses-trajA01; optional, otherwise does all sessions for that trajectory)
subject = 'sub-' + str(sys.argv[1])

if len(sys.argv) > 2:
    traj = str(sys.argv[2])
    session_info = 'ses-traj' + traj
    if len(sys.argv) > 3:
        session_info += str(sys.argv[3])
    else:
        session_info += '*'
else:
    session_info = 'ses-traj*'
    traj = 'A'

to_despike = False  # should be input

# setup directories
# ASSUMES DIRECTORY STRUCTURE:
# ASSUMES atlas ROIs are in: simpace_dir/templates/atlas_name/roi_files (i.e. atlas_name='aal')
simpace_dir = '/home/despo/simpace'
data_dir = pjoin(simpace_dir, 'rename_files')
templates_dir = pjoin(simpace_dir, 'templates')

coreg_prefix = 'r'
anat_dir = 'anat'
func_dir = 'func'
nii_ext = '*.nii'
atlas_name = ['aal', 'power']  # have input to function
workflow_name_anat = 'anat_proc'
workflow_name_func = 'func_proc'

json_dir = '/home/despo/arielle/simpace/simpace'
data_param_file = pjoin(json_dir, 'data_parameters_traj' + traj + '.json')
with open(data_param_file) as fparam:
    dparam = json.load(fparam)
    TR = dparam['TR']
    num_slices = dparam['nb_slices']

# specific directories
subject_dir = pjoin(data_dir, subject)

anat_sess = 'ses-anatomical'
anat_sess_dir = pjoin(subject_dir, anat_sess)
anat_file = glob(pjoin(anat_sess_dir, anat_dir, nii_ext))[0]

workflow_dir = pjoin(data_dir, 'derivatives', subject)
anat_workflow_dir = pjoin(workflow_dir, anat_sess, workflow_name_anat)
coreg_dir = pjoin(anat_workflow_dir, 'coreg_mni_to_native')

sessions = glob(pjoin(subject_dir, session_info))
print subject_dir
print session_info
print sessions


def mv_files(files, dest_dir):
    import shutil
    for ifile in files:
        shutil.move(ifile, dest_dir)


# Set up files for anatomical proc if workflow directory doesn't exist
if not os.path.isdir(anat_workflow_dir):
    TO_run_anatomical = True

    # template brain
    template_file = glob(pjoin(templates_dir, 'mni', 'avg152T1.nii'))[0]

    # atlas files
    for iatlas, atlas in enumerate(atlas_name):
        atlas_roi_files_dir = pjoin(templates_dir, atlas, 'roi_files')
        atlas_roi_files_naming = atlas + nii_ext

        temp_list = glob(pjoin(atlas_roi_files_dir, atlas_roi_files_naming))
        if len(temp_list) == 0:
            temp_list = glob(pjoin(atlas_roi_files_dir, atlas_roi_files_naming) + '.gz')

        temp_list.sort()
        if iatlas == 0:
            atlas_roi_files = temp_list
        else:
            atlas_roi_files = atlas_roi_files + temp_list

    Nrois = len(atlas_roi_files)
    print Nrois

else:
    TO_run_anatomical = False

if TO_run_anatomical:
    segment = pe.Node(interface=spm.Segment(), name='segment')
    segment.inputs.data = anat_file
    segment.inputs.csf_output_type = [False, False, True]  # output native space images
    segment.inputs.gm_output_type = [False, False, True]
    segment.inputs.wm_output_type = [False, False, True]

    wm_file = pe.Node(interface=utility.Rename(), name='wm_file')
    wm_file.inputs.format_string = 'wm_anatres.nii'

    gm_file = pe.Node(interface=utility.Rename(), name='gm_file')
    gm_file.inputs.format_string = 'gm_anatres.nii'

    csf_file = pe.Node(interface=utility.Rename(), name='csf_file')
    csf_file.inputs.format_string = 'csf_anatres.nii'

    # Co-registration of template to subject brain - parameters lifted from
    # http://nipy.org/nipype/interfaces/generated/nipype.interfaces.ants.registration.html
    coreg_mni_to_native = pe.Node(interface=Registration(), name='coreg_mni_to_native')
    coreg_mni_to_native.inputs.fixed_image = anat_file
    coreg_mni_to_native.inputs.moving_image = template_file
    coreg_mni_to_native.inputs.output_transform_prefix = coreg_prefix + "_"
    coreg_mni_to_native.inputs.transforms = ['Affine', 'SyN']
    coreg_mni_to_native.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
    coreg_mni_to_native.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
    coreg_mni_to_native.inputs.dimension = 3
    coreg_mni_to_native.inputs.write_composite_transform = True
    coreg_mni_to_native.inputs.collapse_output_transforms = False
    coreg_mni_to_native.inputs.initialize_transforms_per_stage = False
    coreg_mni_to_native.inputs.metric = ['Mattes'] * 2
    coreg_mni_to_native.inputs.metric_weight = [1] * 2  # Default (value ignored currently by ANTs)
    coreg_mni_to_native.inputs.radius_or_number_of_bins = [32] * 2
    coreg_mni_to_native.inputs.sampling_strategy = ['Random', None]
    coreg_mni_to_native.inputs.sampling_percentage = [0.05, None]
    coreg_mni_to_native.inputs.convergence_threshold = [1.e-8, 1.e-9]
    coreg_mni_to_native.inputs.convergence_window_size = [20] * 2
    coreg_mni_to_native.inputs.smoothing_sigmas = [[1, 0], [2, 1, 0]]
    coreg_mni_to_native.inputs.sigma_units = ['vox'] * 2
    coreg_mni_to_native.inputs.shrink_factors = [[2, 1], [3, 2, 1]]
    coreg_mni_to_native.inputs.use_estimate_learning_rate_once = [True, True]
    coreg_mni_to_native.inputs.use_histogram_matching = [True, True]  # This is the default
    coreg_mni_to_native.inputs.output_warped_image = coreg_prefix + os.path.basename(template_file)

    # Apply co-registration to other files in template space (atlas roi files)
    reg_to_native = pe.MapNode(interface=ApplyTransforms(), name='reg_to_native', iterfield=["input_image"])
    reg_to_native.inputs.input_image = atlas_roi_files
    reg_to_native.inputs.dimension = 3
    reg_to_native.inputs.reference_image = anat_file
    reg_to_native.inputs.interpolation = 'Linear'
    reg_to_native.inputs.invert_transform_flags = [False]
    reg_to_native.inputs.out_postfix = ''

    # Move files since they are separate map node folders
    rename_files = pe.Node(interface=Function(input_names=['files', 'dest_dir'], output_names='out', function=mv_files),
                           name='rename_files')
    rename_files.inputs.dest_dir = coreg_dir

    # set up workflow
    workflow_anatomical = pe.Workflow(name=workflow_name_anat)
    workflow_anatomical.base_dir = pjoin(workflow_dir, anat_sess)

    workflow_anatomical.connect([(segment, gm_file, [('native_gm_image', 'in_file')]),
                                 (segment, wm_file, [('native_wm_image', 'in_file')]),
                                 (segment, csf_file, [('native_csf_image', 'in_file')]),
                                 (coreg_mni_to_native, reg_to_native, [('composite_transform', 'transforms')]),
                                 (reg_to_native, rename_files, [('output_image', 'files')])])

    workflow_anatomical.run()

if len(sessions) > 0:

    # Collect files that are in native space to push into functional space

    # atlas files
    for iatlas, atlas in enumerate(atlas_name):
        atlas_roi_files_naming = atlas + nii_ext

        gz_list = glob(pjoin(coreg_dir, atlas_roi_files_naming + '.gz'))
        for ifile in gz_list:
            cmd = 'gunzip ' + ifile
            os.system(cmd)

        temp_list = glob(pjoin(coreg_dir, atlas_roi_files_naming))
        temp_list.sort()
        if iatlas == 0:
            atlas_files = temp_list
        else:
            atlas_files = atlas_files + temp_list

    Nrois = len(atlas_files)

    files_to_coreg_to_native = [glob(pjoin(anat_workflow_dir, 'gm_file', nii_ext))[0],
                                glob(pjoin(anat_workflow_dir, 'wm_file', nii_ext))[0],
                                glob(pjoin(anat_workflow_dir, 'csf_file', nii_ext))[0]]
    files_to_coreg_to_native.extend(atlas_files)
    print len(files_to_coreg_to_native)

# Pre-processing workflow for functional data
for isess, sess_dir in enumerate(sessions):
    sess = os.path.basename(sess_dir)

    func_workflow_dir = pjoin(workflow_dir, sess, workflow_name_func)

    workflow_func = pe.Workflow(name=workflow_name_func)
    workflow_func.base_dir = pjoin(workflow_dir, sess)

    # for naming directories for de-spiked data
    if to_despike:
        name_append = '_despike'
    else:
        name_append = ''

    # grab input files
    # run_folders = glob(pjoin(sess_dir, func_dir, run_str))
    input_files = glob(pjoin(sess_dir, func_dir, nii_ext))
    # for irun in run_folders:
    #     input_files.append(glob(pjoin(irun, nifti_dir, os.path.basename(irun)) + nii_ext[1:len(nii_ext)])[0])

    stc = pe.Node(interface=spm.SliceTiming(), name='stc' + name_append)
    stc.inputs.num_slices = num_slices
    stc.inputs.time_repetition = TR
    stc.inputs.time_acquisition = TR - TR / num_slices
    stc.inputs.slice_order = range(num_slices, 0, -1)  # slices are descending
    stc.inputs.ref_slice = num_slices/2  # close to middle slice in time

    realign = pe.Node(interface=spm.Realign(), name='realign' + name_append)
    realign.inputs.register_to_mean = True

    smooth = pe.Node(interface=spm.Smooth(), name='smooth' + name_append)
    smooth.inputs.fwhm = [6, 6, 6]

    motion_params = pe.MapNode(interface=utility.Rename(), name='motion_params' + name_append, iterfield=["in_file"])
    motion_params.inputs.format_string = 'MP_%(run_name)s' + name_append + '.txt'
    motion_params.inputs.parse_string = 'rp_.*_.*_acq-(?P<run_name>\w*)_bold.txt'

    motion_params_movefiles = pe.Node(interface=Function(input_names=['files', 'dest_dir'], output_names='out',
                                                         function=mv_files),
                                      name='motion_params_movefiles' + name_append)
    motion_params_movefiles.inputs.dest_dir = pjoin(func_workflow_dir, 'motion_params')

    if to_despike:
        despike = pe.MapNode(interface=afni.Despike(), name='despike', iterfield=["in_file"])
        despike.inputs.in_file = input_files
        despike.inputs.outputtype = 'NIFTI'
        workflow_func.connect([(despike, stc, [('out_file', 'in_files')])])

    else:
        stc.inputs.in_files = input_files

    workflow_func.connect([(stc, realign, [('timecorrected_files', 'in_files')]),
                           (realign, smooth, [('realigned_files', 'in_files')]),
                           (realign, motion_params, [('realignment_parameters', 'in_file')]),
                           (motion_params, motion_params_movefiles, [('out_file', 'files')])
                           ])

    # add in co-registration of anatomical to functional
    if not to_despike:
        coreg = pe.Node(interface=spm.Coregister(), name='coreg')
        coreg.inputs.source = anat_file
        coreg.inputs.apply_to_files = files_to_coreg_to_native  # output of segment & roi_files from convert_mni_to_native

        select_gm = pe.Node(interface=utility.Select(), name='select_gm')
        select_gm.inputs.set(index=[0])  # selects gm

        select_wm = pe.Node(interface=utility.Select(), name='select_wm')
        select_wm.inputs.set(index=[1])  # selects wm

        select_csf = pe.Node(interface=utility.Select(), name='select_csf')
        select_csf.inputs.set(index=[2])  # selects csf

        # split & rename for aal outputs from coreg 4:(99+4)
        select_roi_files = pe.Node(interface=utility.Select(), name='select_roi_files')
        select_roi_files.inputs.set(index=range(3, 3 + Nrois))

        anat_in_func_res = pe.Node(interface=utility.Rename(), name='anat_in_func_res')
        anat_in_func_res.inputs.format_string = 'T1_func_res.nii'

        csf_in_func_res = pe.Node(interface=utility.Rename(), name='csf_in_func_res')
        csf_in_func_res.inputs.format_string = 'csf_func_res.nii'

        wm_in_func_res = pe.Node(interface=utility.Rename(), name='wm_in_func_res')
        wm_in_func_res.inputs.format_string = 'wm_func_res.nii'

        gm_in_func_res = pe.Node(interface=utility.Rename(), name='gm_in_func_res')
        gm_in_func_res.inputs.format_string = 'gm_func_res.nii'

        workflow_func.connect([(realign, coreg, [('mean_image', 'target')]),
                               (coreg, select_csf, [('coregistered_files', 'inlist')]),
                               (coreg, select_wm, [('coregistered_files', 'inlist')]),
                               (coreg, select_gm, [('coregistered_files', 'inlist')]),
                               (coreg, select_roi_files, [('coregistered_files', 'inlist')]),
                               (coreg, anat_in_func_res, [('coregistered_source', 'in_file')]),
                               (select_csf, csf_in_func_res, [('out', 'in_file')]),  # NEED?
                               (select_wm, wm_in_func_res, [('out', 'in_file')]),
                               (select_gm, gm_in_func_res, [('out', 'in_file')]),
                               ])

    workflow_func.run()
    os.system(pjoin('gzip ' + func_workflow_dir, 'stc', 'a*.nii'))
    os.system(pjoin('gzip ' + func_workflow_dir, 'realign', 'ra*.nii'))

    if not to_despike:
        os.system('python Pmaps_v3.py ' + func_workflow_dir)
