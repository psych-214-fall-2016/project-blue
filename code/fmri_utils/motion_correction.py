""" Functions for motion correcton and spatial normalization
    between individuals or individual to template.

"""
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from scipy.optimize import fmin_powell

#: gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'

project_path = '../../'
data_path = project_path+'data/sub_data/'

#We have 26 dataset
#change here to get your subject !
# subject_list = [str(i) for i in range(1,27)]
image_path = data_path + 'func/sub-01_task-visualimageryfalsememory_run-01_bold.nii'


def load_data():
    """ Load the volume data for finding the optimized transformation matrix.
    """

    #load a 4 D data for individual subject
    img = nib.load(image_path)
    data = img.get_data()

    #we should just collect the data after the outlier detection
    #1 will be replace with the outliers
    fixed_data = data[...,1:]

    vol1 = data[..., 0]

    #vol2 can be any volume other than the first volume we want to align to
    vol2 = data[..., 1]

    return vol1, vol2

def cost_function(matvec):
    """ Give cost function value at each affine transformation.
    """

    #mat is the rotation and scaling, vec is the translation
    mat, vec = nib.affines.to_matvec(matvec)

    vol1, vol2 = load_data()
    # order=1 for linear interpolation
    #The output_shape is supposed to be same
    vol1_transformed = affine_transform(vol1, mat, vec, order=1)

    """ Negative correlation between the two volumes, flattened to 1D """
    correl = np.corrcoef(vol1.ravel(), vol2.ravel())[0, 1]
    return -correl

def transformation_rigid_matrix(p):
    """transformation_rigid_matrix returns a matrix defining a rigid-body transformation.
    The transformation can be a translation, rotation, scaling.
    (well scalings are not rigid, but it's customary to include
    them in the model) transformation given a vector of parameters (p).
    By default, the transformations are applied in the following order (i.e.,
    the opposite to which they are specified):
    1) scale (zoom)
    2) rotation - yaw, roll & pitch
    3) translation
    Parameters
    ----------
    p: array_like of length 9
        vector of parameters
        p(0)  - x translation
        p(1)  - y translation
        p(2)  - z translation
        p(3)  - x rotation about - {pitch} (radians)
        p(4)  - y rotation about - {roll}  (radians)
        p(5)  - z rotation about - {yaw}   (radians)
        p(6)  - x scaling
        p(7)  - y scaling
        p(8)  - z scaling
    Returns
    -------
    A: 2D array of shape (4, 4)
        orthogonal transformation matrix
    """

    # get initial parameters
    q = np.zeros(9) + 2
    q[6:9] = 1.
     # fill-up up p to length 9, if p is too short
    p = np.hstack((p, q[len(p):12]))

    # translation
    T = np.eye(4)
    T[:3, -1] = p[:3]

    # yaw
    R1 = np.eye(4)
    R1[1, 1:3] = np.cos(p[3]), np.sin(p[3])
    R1[2, 1:3] = -np.sin(p[3]), np.cos(p[3])

    # roll
    R2 = np.eye(4)
    R2[0, [0, 2]] = np.cos(p[4]), np.sin(p[4])
    R2[2, [0, 2]] = -np.sin(p[4]), np.cos(p[4])

    # pitch
    R3 = np.eye(4)
    R3[0, 0:2] = np.cos(p[5]), np.sin(p[5])
    R3[1, 0:2] = -np.sin(p[5]), np.cos(p[5])

    # rotation
    R = np.dot(R1, np.dot(R2, R3))

    # zoom & scaling
    Z = np.eye(4)
    np.fill_diagonal(Z[0:3, 0:3], p[6:9])

    # scaling
    S = np.eye(4)
    S[0, 1:3] = p[9:11]
    S[1, 2] = p[11]

    # compute the complete affine transformation
    M = np.dot(T, np.dot(R, Z))

    best_params = fmin_powell(cost_function, [0, 0, 0, 0, 0, 0, 1, 1, 1], callback=my_callback)

    return best_params
