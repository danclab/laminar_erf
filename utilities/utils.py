import json
import math
import os
import os.path as op
import nibabel as nib
import numpy as np
import pandas as pd
from lameg.surf import mesh_adjacency
from scipy import interpolate
from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components
from scipy.signal import correlate


def get_subject_sessions_idx():
    subjects=[]
    sessions=[]
    for sub_idx in range(8):
        if sub_idx==0:
            subjects.append(sub_idx)
            sessions.append(0)
        elif sub_idx>=1 and sub_idx<=4:
            for ses_idx in range(4):
                subjects.append(sub_idx)
                sessions.append(ses_idx)
        elif sub_idx==5:
            subjects.append(sub_idx)
            sessions.append(0)
        elif sub_idx==6:
            for ses_idx in range(4):
                subjects.append(sub_idx)
                sessions.append(ses_idx)
        elif sub_idx==7:
            for ses_idx in range(3):
                subjects.append(sub_idx)
                sessions.append(ses_idx)
    return subjects, sessions

def align_signals(signal, reference_signal):
    """
    Aligns a signal to a reference signal using cross-correlation, without truncation or improper wrapping.

    :param signal: The signal to be aligned.
    :param reference_signal: The reference signal to align to.
    :return: Aligned signal.
    """
    # Compute cross-correlation
    correlation = correlate(reference_signal, signal, mode='full')

    # Find the shift that maximizes the cross-correlation
    shift = np.argmax(correlation) - len(reference_signal) + 1

    # Apply the shift to align the signal
    if shift > 0:
        aligned_signal = np.concatenate([np.zeros(shift), signal])[:len(reference_signal)]
    else:
        aligned_signal = np.concatenate([signal, np.zeros(-shift)])[-len(reference_signal):]

    return aligned_signal


def woody(signals):
    """
    Computes the weighted average of signals using the Woody method.

    :param signals: A list of signals (each a numpy array) to be averaged.
    :param weights: Corresponding weights for each signal.
    :return: The weighted average signal.
    """
    reference_signal = np.mean(signals, axis=0)
    aligned_signals = np.vstack([align_signals(signal, reference_signal) for signal in signals])

    return aligned_signals


def get_roi_idx(subj_id, surf_dir, hemi, regions, surf):
    fs_subjects_dir = os.getenv('SUBJECTS_DIR')
    fs_subject_dir = os.path.join(fs_subjects_dir, subj_id)

    roi_idx = []
    hemis = []
    if hemi is None:
        hemis.extend(['lh', 'rh'])
    else:
        hemis.append(hemi)
    for hemi in hemis:
        pial = nib.load(os.path.join(surf_dir, f'{hemi}.pial.gii'))

        annotation = os.path.join(fs_subject_dir, 'label', f'{hemi}.aparc.annot')
        label, ctab, names = nib.freesurfer.read_annot(annotation)

        name_indices = [names.index(region.encode()) for region in regions]
        orig_vts = np.where(np.isin(label, name_indices))[0]

        # Find the original vertices closest to the downsampled vertices
        kdtree = KDTree(pial.darrays[0].data[orig_vts, :])
        # Calculate the percentage of vertices retained
        dist, vert_idx = kdtree.query(surf.darrays[0].data, k=1)
        hemi_roi_idx = np.where(dist == 0)[0]
        roi_idx = np.union1d(roi_idx, hemi_roi_idx)
    return roi_idx.astype(int)


def smooth_signal(def_df, threshold=10, proximity=5):
    """
    Smooth out abrupt changes in a signal where the derivative exceeds the specified threshold.
    Also smooth out points that are within a certain proximity of abrupt changes.

    :param def_df: Input signal array.
    :param threshold: Threshold for detecting abrupt changes in the derivative (default is 10).
    :param proximity: Number of points surrounding an abrupt change to be considered for smoothing.
    :return: Smoothed signal.
    """
    # Compute the derivative
    derivative = np.diff(def_df)

    # Initialize an array to mark points for smoothing
    should_smooth = np.zeros(len(def_df), dtype=bool)

    # Find indices where the change is too abrupt
    abrupt_changes = np.where((np.abs(derivative) > threshold))[0] + 1  # Offset by one due to np.diff()

    # Mark points that are within the proximity of abrupt changes
    for index in abrupt_changes:
        start_idx = max(0, index - proximity)
        end_idx = min(len(def_df), index + proximity + 1)
        should_smooth[start_idx:end_idx] = True

    # Handle points for smoothing
    if not should_smooth.any():
        return def_df

    # Prepare for interpolation
    x = np.arange(len(def_df))
    y = def_df.copy()
    good_points = ~should_smooth

    # Interpolate to smooth out marked points
    f = interpolate.interp1d(x[good_points], y[good_points], kind='linear', bounds_error=False,
                             fill_value="extrapolate")

    # Apply the interpolation to smooth the signal
    y[should_smooth] = f(x[should_smooth])

    return y


def find_clusters(faces, threshold_indices):
    # Use the provided function to compute the adjacency matrix
    adjacency_matrix = mesh_adjacency(faces)

    # Convert the adjacency matrix to CSR format for efficient slicing
    adjacency_matrix_csr = adjacency_matrix.tocsr()

    # Focus on the subgraph of threshold-exceeding vertices
    subgraph_adjacency_matrix = adjacency_matrix_csr[threshold_indices, :][:, threshold_indices]

    # Find connected components in the subgraph
    n_components, labels = connected_components(csgraph=subgraph_adjacency_matrix, directed=False, return_labels=True)

    # Organize vertices into clusters based on their component labels
    clusters = []
    for i in range(n_components):
        cluster_indices = np.where(labels == i)[0]
        clusters.append([threshold_indices[idx] for idx in cluster_indices])

    return clusters


def get_fiducial_coords(subj_id, json_file):
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]
    df = pd.read_csv(op.join(path, 'raw/participants.tsv'), sep='\t')

    # Fetch the row corresponding to the given subj_id
    row = df.loc[df['subj_id'] == subj_id]
    # Parse the 'nas', 'lpa', and 'rpa' columns into lists
    nas = row['nas'].values[0].split(',')
    nas = [float(i) for i in nas]
    lpa = row['lpa'].values[0].split(',')
    lpa = [float(i) for i in lpa]
    rpa = row['rpa'].values[0].split(',')
    rpa = [float(i) for i in rpa]
    return lpa, nas, rpa


def gaussian(x, mu, sig, amp):
    """
    Return gaussian on a x axis.

    Args:
        x (numpy.ndarray): axis (preferably a product of a np.linspace)
        mu (float): location on x axis
        sig (float): spread
        amp (float): amplitude
    """
    gauss = 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    mn, mx = gauss.min(), gauss.max()
    gauss = (gauss - mn) / (mx - mn)
    gauss = gauss * amp
    return gauss


def rotate_point(point, pitch, roll, yaw):
    """
    Rotate a 3D point around the origin.

    Args:
        point (numpy.ndarray): The 3D point to be rotated.
        pitch (float): The rotation angle around the x-axis in degrees.
        roll (float): The rotation angle around the y-axis in degrees.
        yaw (float): The rotation angle around the z-axis in degrees.

    Returns:
        numpy.ndarray: The rotated 3D point.
    """
    # Convert angles to radians
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)
    yaw_rad = math.radians(yaw)

    # Define rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
                   [0, math.sin(pitch_rad), math.cos(pitch_rad)]])
    Ry = np.array([[math.cos(roll_rad), 0, math.sin(roll_rad)],
                   [0, 1, 0],
                   [-math.sin(roll_rad), 0, math.cos(roll_rad)]])
    Rz = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad), 0],
                   [math.sin(yaw_rad), math.cos(yaw_rad), 0],
                   [0, 0, 1]])

    # Perform rotations
    rotated_point = np.dot(Rz, np.dot(Ry, np.dot(Rx, point)))

    return rotated_point


def rigid_body_transform(tip, left, right, pitch, roll, yaw, x_trans, y_trans, z_trans):
    """
    Perform a rigid body transformation on a triangle.

    Args:
        tip (numpy.ndarray): The 3D point representing the tip of the triangle.
        left (numpy.ndarray): The 3D point representing the left corner of the triangle.
        right (numpy.ndarray): The 3D point representing the right corner of the triangle.
        pitch (float): The rotation angle around the x-axis in degrees.
        roll (float): The rotation angle around the y-axis in degrees.
        yaw (float): The rotation angle around the z-axis in degrees.
        x_trans (float): The translation along the x-axis.
        y_trans (float): The translation along the y-axis.
        z_trans (float): The translation along the z-axis.

    Returns:
        tuple: A tuple containing the transformed 3D points of the triangle (tip, left, right).
    """
    # Rotate points
    tip_rotated = rotate_point(tip, pitch, roll, yaw)
    left_rotated = rotate_point(left, pitch, roll, yaw)
    right_rotated = rotate_point(right, pitch, roll, yaw)

    # Translate points
    tip_translated = tip_rotated + np.array([x_trans, y_trans, z_trans])
    left_translated = left_rotated + np.array([x_trans, y_trans, z_trans])
    right_translated = right_rotated + np.array([x_trans, y_trans, z_trans])

    return tip_translated, left_translated, right_translated