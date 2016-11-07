
from os.path import join as pjoin
from glob import glob
import nipype.pipeline.engine as pe
import nipype.interfaces.spm as spm
import nipype.interfaces.matlab as matlab
import nipype.interfaces.fsl as fsl
from nipype.interfaces.ants.segmentation import BrainExtraction
from nipype.interfaces.ants.legacy import antsIntroduction
from nipype.interfaces.ants import WarpImageMultiTransform
from nipype.interfaces import utility
import nipype.interfaces.io as nio
import json, os

matlab.MatlabCommand.set_default_paths('/usr/local/matlab-tools/spm/spm8')
os.environ['SGE_ROOT'] = ''

data_dir = '/home/despo/arielle/hp-tms-mri/data'
nifti_dir = pjoin(data_dir, 'nifti-bids')
ss = '206'
session = 'baseline'

smth_fwhm = 6

ants_template = '/home/despo/arielle/T_template0.nii.gz'
ants_template_mask = '/home/despo/arielle/T_template0ProbabilityMask.nii.gz'

sub = 'sub-' + ss
sess = 'ses-baseline'
sess_dir = glob(pjoin(nifti_dir, sub, sess))[0]
workflow_dir = pjoin(data_dir, 'workflows', sub, sess)
sub_dir = pjoin(data_dir, 'workflows', sub)
anat_dir = pjoin(sub_dir, 'anat_files')

anat_file_bias = glob(pjoin(anat_dir, 'm*_spm.nii'))[0]
# anat_file = glob(pjoin(sess_dir, 'anat', 's*_spm.nii'))[0]
func_files = glob(pjoin(sess_dir, 'func', '*.nii'))

info = json.load(open(os.path.splitext(func_files[0])[0] + '.json'))
num_slices = len(info['SliceTiming'])
TR = info['RepetitionTime']
# functional workflow

# segment = pe.Node(interface=spm.Segment(), name='segment')
# segment.inputs.data = anat_file
# segment.inputs.csf_output_type = [False, False, True]  # output native space images
# segment.inputs.gm_output_type = [False, False, True]
# segment.inputs.wm_output_type = [False, False, True]

brainextraction = pe.Node(interface=BrainExtraction(), name='brainextraction')
brainextraction.inputs.dimension = 3
brainextraction.inputs.anatomical_image = anat_file_bias
brainextraction.inputs.brain_template = ants_template
brainextraction.inputs.brain_probability_mask = ants_template_mask

antsreg = pe.Node(interface=antsIntroduction(), name='antsreg')
antsreg.inputs.reference_image = "/home/despo/arielle/hp-tms/ants/ants-template.nii.gz"

sphere_name = "_sphere_15mm.nii.gz"
ants_sphere = "/home/despo/arielle/hp-tms/seed_analysis_at/Mid_R_ROIfiles/ants" + sphere_name
antswarp = pe.Node(interface=WarpImageMultiTransform(), name='antswarp')
antswarp.inputs.input_image = ants_sphere
# antswarp.inputs.transformation_series = []
antswarp.inputs.invert_affine = [1]
# antswarp.inputs.output_image = 'native' + sphere_name
antswarp.inputs.out_postfix = '_native'

struct_list = ['R_Hipp', 'L_Hipp', 'R_Amyg', 'L_Amyg']
first_be = pe.Node(interface=fsl.FIRST(), name='first_be')
first_be.inputs.brain_extracted = True
# first_be.inputs.list_of_specific_structures = struct_list

stc = pe.Node(interface=spm.SliceTiming(), name='stc')
stc.inputs.num_slices = num_slices
stc.inputs.time_repetition = TR
stc.inputs.time_acquisition = TR - TR / num_slices
stc.inputs.slice_order = range(num_slices, 0, -1)  # slices are descending
stc.inputs.ref_slice = num_slices/2  # close to middle slice in time
stc.inputs.in_files = func_files

realign = pe.Node(interface=spm.Realign(), name='realign')
realign.inputs.register_to_mean = True

smooth = pe.Node(interface=spm.Smooth(), name='smooth')
smooth.inputs.fwhm = [smth_fwhm, smth_fwhm, smth_fwhm]

wm_file = pe.Node(interface=utility.Rename(), name='wm_file')
wm_file.inputs.format_string = 'wm_anatres.nii'

gm_file = pe.Node(interface=utility.Rename(), name='gm_file')
gm_file.inputs.format_string = 'gm_anatres.nii'

csf_file = pe.Node(interface=utility.Rename(), name='csf_file')
csf_file.inputs.format_string = 'csf_anatres.nii'

coreg = pe.Node(interface=spm.Coregister(), name='coreg')
coreg.inputs.target = anat_file_bias
coreg.inputs.jobtype = 'estimate'

workflow = pe.Workflow(name='proc_vspm')
workflow.base_dir = workflow_dir

datasink = pe.Node(nio.DataSink(), name='sinker')
datasink.inputs.base_directory = pjoin(workflow_dir)

datasink_tms = pe.Node(nio.DataSink(), name='sinker_tms')
datasink_tms.inputs.base_directory = pjoin(sub_dir)

merge_xfms = pe.Node(utility.Merge(2), name='merge_xfms')
# merge_xfms.inputs.

print os.system('echo $ANTSPATH')
#os.chdir(pjoin(workflow_dir, 'proc_vspm', 'first_be'))

workflow.connect([(stc, realign, [('timecorrected_files', 'in_files')]),
                (realign, smooth, [('realigned_files', 'in_files')]),
                (realign, coreg, [('mean_image', 'source')]),
                (realign, datasink, [('realignment_parameters', 'motion_files')]),
                (smooth, coreg, [('smoothed_files', 'apply_to_files')]),
                (brainextraction, datasink, [('BrainExtractionBrain', 'ants_brain_extract')]),
                (brainextraction, first_be, [('BrainExtractionBrain', 'in_file')]),
                (brainextraction, antsreg, [('BrainExtractionBrain', 'input_image')]),
                (brainextraction, antswarp, [('BrainExtractionBrain', 'reference_image')]),
                (antsreg, merge_xfms, [('affine_transformation', 'in1')]),
                (antsreg, merge_xfms, [('inverse_warp_field', 'in2')]),
                (merge_xfms, antswarp, [('out', 'transformation_series')]),
                (antswarp, datasink_tms, [('output_image', 'tms_files')]),
                (antsreg, datasink, [('warp_field', 'ants_to_template_files')]),
                (merge_xfms, datasink, [('out', 'ants_to_template_files.@xfms')])
                # (first_be, datasink, [('original_segmentations', 'first_be')])
                ])




# workflow.config['execution'] = {'remove_unnecessary_outputs': False}
workflow.run()


