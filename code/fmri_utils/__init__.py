 # Init for utils package
from .spm_funcs import spm_global, get_spm_globals, spm_hrf
from .model_signal import data_timecourse, create_design_matrix, event_timecourse, beta_res_calc, compute_tstats
from .detectors import mean_detector, std_detector, iqr_detector
from .motion_correction import create_rotmat, transformation_matrix, apply_transform, cost_function, optimize_params
from .dir_utils import search_directory, get_contents, file_hash, validate_data
from .slice_timing_corr import slice_timing_corr
