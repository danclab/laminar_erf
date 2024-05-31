import json
import os
import os.path as op
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial import KDTree
from lameg.surf import compute_mesh_adjacency, compute_geodesic_distances
from scipy.sparse.csgraph import connected_components
from scipy.signal import correlate

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
    adjacency_matrix = compute_mesh_adjacency(faces)

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


def parallel_compute_distances(vtx_index, distance_matrix, vtx_to_plot):
    """
    Compute geodesic distances from a single source vertex to all vertices in vtx_to_plot.

    :param vtx_index: Index of the source vertex in vtx_to_plot.
    :param distance_matrix: Precomputed distance matrix or similar structure.
    :param vtx_to_plot: List of vertex indices for which to compute distances.
    :return: Array of geodesic distances from the source vertex to each vertex in vtx_to_plot.
    """
    all_dist = compute_geodesic_distances(distance_matrix, [vtx_to_plot[vtx_index]])
    return all_dist[vtx_to_plot]


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