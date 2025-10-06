import os
import shutil

import mne
import nibabel as nib
import numpy as np
from lameg.invert import coregister, invert_ebb, load_source_time_series
from lameg.laminar import sliding_window_model_comparison
from lameg.util import get_fiducial_coords, get_surface_names
import spm_standalone
import matlab
import h5py
from mne import read_epochs
from scipy.spatial import cKDTree, KDTree
from scipy.stats import zscore


def get_roi_idx(subj_id, surf_dir, hemi, regions, surf):
    """
    Returns the vertex indices from a downsampled surface mesh that correspond
    to specified anatomical regions (from FreeSurfer annotations).

    Parameters:
    - subj_id (str): FreeSurfer subject ID.
    - surf_dir (str): Path to directory containing the postprocessed surfaces (e.g., pial.gii).
    - hemi (str or None): Hemisphere ('lh', 'rh', or None for both).
    - regions (list of str): List of region names to extract from the annotation.
    - surf (nib.GiftiImage): Downsampled surface mesh used for laminar inference.

    Returns:
    - roi_idx (np.ndarray): Array of indices into `surf` corresponding to the selected ROI.
    """
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


def get_cortical_thickness(multilayer_mesh, n_layers):
    """
    Estimates cortical thickness at each vertex from a multilayer surface mesh.

    Parameters:
    - multilayer_mesh (nib.GiftiImage): Combined mesh of all depth layers (equidistant).
    - n_layers (int): Total number of cortical depth surfaces in the mesh.

    Returns:
    - thickness (np.ndarray): Vertex-wise Euclidean distance between layer 1 and layer N.
    """
    verts_per_surf = int(multilayer_mesh.darrays[0].data.shape[0] / n_layers)
    thickness = np.sqrt(np.sum((multilayer_mesh.darrays[0].data[:verts_per_surf, :] - multilayer_mesh.darrays[0].data[
                                                                                      (n_layers - 1) * verts_per_surf:,
                                                                                      :]) ** 2, axis=-1))
    return thickness


def get_lead_field_rmse(gainmat_fname, n_layers, verts_per_surf):
    """
    Computes the RMSE of lead field vectors across cortical depths
    relative to the superficial (layer 1) model, for each vertex.

    Parameters:
    - gainmat_fname (str): Path to HDF5 file containing the gain matrix ('G').
    - n_layers (int): Number of cortical depth layers.
    - verts_per_surf (int): Number of vertices per layer.

    Returns:
    - rmse (np.ndarray): RMSE between deep and superficial lead fields per vertex.
    """
    with h5py.File(gainmat_fname, 'r') as file:
        lf_mat = file['G'][()]

    layer_lf_mat = np.zeros((verts_per_surf, n_layers, lf_mat.shape[1]))
    diff_layer_lf_mat = np.zeros((verts_per_surf, n_layers))
    for i in range(n_layers):
        layer_lf_mat[:, i, :] = lf_mat[i * verts_per_surf:(i + 1) * verts_per_surf, :]
    for j in range(verts_per_surf):
        for i in range(n_layers):
            diff_layer_lf_mat[j, i] = np.sqrt(np.mean((layer_lf_mat[j, i, :] - layer_lf_mat[j, 0, :]) ** 2))
    return diff_layer_lf_mat[:, -1]


def get_orientation(scalp_mesh_fname, multilayer_mesh, pial_ds_mesh, verts_per_surf):
    """
    Computes the alignment between cortical column orientation vectors and
    local scalp surface normals for each vertex.

    Parameters:
    - scalp_mesh_fname (str): Path to scalp surface (GIFTI).
    - multilayer_mesh (nib.GiftiImage): Mesh with orientation vectors per vertex.
    - pial_ds_mesh (nib.GiftiImage): Downsampled pial mesh for mapping.
    - verts_per_surf (int): Number of vertices per cortical surface.

    Returns:
    - dot_products (np.ndarray): Cosine similarity (abs dot product) between
                                 dipole and scalp normals per vertex.

    Notes:
    - Higher values indicate more radial orientations
    """
    scalp_mesh = nib.load(scalp_mesh_fname)
    scalp_vertices = scalp_mesh.darrays[1].data  # Vertex coordinates
    scalp_faces = scalp_mesh.darrays[0].data  # Face indices

    pial_ds_vertices = pial_ds_mesh.darrays[0].data  # Vertex coordinates
    multilayer_orientations = multilayer_mesh.darrays[2].data  # Orientation vectors

    # Compute surface normals
    def compute_normals(vertices, faces):
        normals = np.zeros_like(vertices)
        for i in range(faces.shape[0]):
            v0, v1, v2 = vertices[faces[i]]
            # Edge vectors
            edge1 = v1 - v0
            edge2 = v2 - v0
            # Cross product to get normal
            normal = np.cross(edge1, edge2)
            normal /= np.linalg.norm(normal)  # Normalize the normal
            normals[faces[i]] += normal
        normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize all normals
        return normals

    normals = compute_normals(scalp_vertices, scalp_faces)

    # Build a KD-tree for the scalp vertices
    tree = cKDTree(scalp_vertices)

    # Find the nearest neighbor in the scalp mesh for each vertex in the downsampled cortical mesh
    _, indices = tree.query(pial_ds_vertices)

    # Compute the dot product between the orientation and the normal vectors
    dot_products = np.abs(np.einsum('ij,ij->i', normals[indices], multilayer_orientations[:verts_per_surf, :]))
    return dot_products


def get_dist_to_scalp(scalp_mesh_fname, pial_ds_mesh):
    """
    Computes the Euclidean distance from each vertex on the cortical surface
    to the closest point on the scalp.

    Parameters:
    - scalp_mesh_fname (str): Path to scalp surface (GIFTI).
    - pial_ds_mesh (nib.GiftiImage): Downsampled pial surface.

    Returns:
    - distances (np.ndarray): Distance (in mm) from cortex to nearest scalp point per vertex.
    """
    scalp_mesh = nib.load(scalp_mesh_fname)
    scalp_vertices = scalp_mesh.darrays[1].data  # Vertex coordinates

    pial_ds_vertices = pial_ds_mesh.darrays[0].data  # Vertex coordinates

    # Build a KD-tree for the scalp vertices
    tree = cKDTree(scalp_vertices)

    # Find the nearest neighbor in the scalp mesh for each vertex in the downsampled cortical mesh
    distances, _ = tree.query(pial_ds_vertices)

    return distances


subjects = [
    'sub-001',
    'sub-002', 'sub-002', 'sub-002', 'sub-002', 'sub-002', 'sub-002',
    'sub-003', 'sub-003', 'sub-003', 'sub-003',
    'sub-004', 'sub-004', 'sub-004', 'sub-004',
    'sub-005', 'sub-005', 'sub-005', 'sub-005',
    'sub-006', 'sub-006', 'sub-006',
    'sub-007', 'sub-007', 'sub-007', 'sub-007',
    'sub-008', 'sub-008', 'sub-008', 'sub-008', 'sub-008'
]
sessions = [
    'ses-01',
    'ses-01', 'ses-02', 'ses-03', 'ses-04', 'ses-05', 'ses-06',
    'ses-01', 'ses-02', 'ses-03', 'ses-04',
    'ses-01', 'ses-02', 'ses-03', 'ses-04',
    'ses-01', 'ses-02', 'ses-03', 'ses-04',
    'ses-01', 'ses-02', 'ses-03',
    'ses-01', 'ses-02', 'ses-03', 'ses-04',
    'ses-01', 'ses-02', 'ses-03', 'ses-04', 'ses-05'
]

subjects_file = '/home/common/bonaiuto/cued_action_meg/raw/participants.tsv'
preproc_data_dir = '/home/common/bonaiuto/cued_action_meg/derivatives/processed'

n_layers = 11


def run(patch_size, spm):
    base_out_dir = os.path.join('/home/bonaiuto/laminar_erf/output/motor_epoch', '01_mf', f'patch_size_{patch_size:.2f}mm', 'noautoreject')
    if not os.path.exists(base_out_dir):
        os.mkdir(base_out_dir)

    for subject, session in zip(subjects, sessions):
        print(f'{subject}: {session}')
        sess_path = os.path.join(preproc_data_dir, subject, session)
        data_dir = os.path.join(sess_path, 'spm')
        out_dir = os.path.join(base_out_dir, subject, session)
        if not os.path.exists(os.path.join(base_out_dir, subject)):
            os.mkdir(os.path.join(base_out_dir, subject))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        out_fname = os.path.join(base_out_dir, f'results_{subject}_{session}.npz')

        if True:#not os.path.exists(out_fname):
            mri_fname = os.path.join(preproc_data_dir, subject, 't1w.nii')

            subj_surf_dir = os.path.join(preproc_data_dir, subject, 'surf')
            multilayer_mesh_fname = os.path.join(subj_surf_dir, 'multilayer.11.ds.link_vector.fixed.gii')
            multilayer_mesh = nib.load(multilayer_mesh_fname)
            verts_per_surf = int(multilayer_mesh.darrays[0].data.shape[0] / n_layers)
            ds_pial = nib.load(os.path.join(subj_surf_dir, 'pial.ds.gii'))
            layer_fnames = get_surface_names(
                n_layers,
                subj_surf_dir,
                'link_vector.fixed'
            )

            nas, lpa, rpa = get_fiducial_coords(subject, subjects_file)

            #orig_data_file = f'spm_converted_autoreject-{subject}-{session}-motor-epo-ave.mat'
            orig_data_file = f'spm_converted_{subject}-{session}-motor-epo-ave.mat'
            data_base = os.path.splitext(orig_data_file)[0]

            # Copy data files to tmp directory
            shutil.copy(
                os.path.join(data_dir, f'{data_base}.mat'),
                os.path.join(out_dir, f'{data_base}.mat')
            )
            shutil.copy(
                os.path.join(data_dir, f'{data_base}.dat'),
                os.path.join(out_dir, f'{data_base}.dat')
            )

            # Construct base file name for simulations
            base_fname = os.path.join(out_dir, f'{data_base}.mat')

            #epochs = read_epochs(os.path.join(sess_path, f'autoreject-{subject}-{session}-motor-epo.fif'),
            epochs = read_epochs(os.path.join(sess_path, f'{subject}-{session}-motor-epo.fif'),
                                 verbose=False, preload=True)
            rank = mne.compute_rank(epochs, rank='info')

            # Coregister data to multilayer mesh
            coregister(
                nas,
                lpa,
                rpa,
                mri_fname,
                multilayer_mesh_fname,
                base_fname,
                spm_instance=spm,
                viz=True
            )

            # Run localizer on multilayer mesh on -250 to 250ms time window
            [_, _, MU] = invert_ebb(
                multilayer_mesh_fname,
                base_fname,
                n_layers,
                woi=[-75+2000, -25+2000],
                patch_size=patch_size,
                n_temp_modes=4,
                n_spatial_modes=rank['mag'],
                return_mu_matrix=True,
                spm_instance=spm,
                viz=True
            )

            # Get left precentral vertices
            roi_idx = get_roi_idx(subject, subj_surf_dir, 'lh', ['precentral'], ds_pial)

            # Find max variance in -100 to 25ms time window in the middle surface layer
            all_layer_ts, ts_time, _ = load_source_time_series(
                base_fname,
                mu_matrix=MU,
                vertices=(5 - 1) * verts_per_surf + np.arange(verts_per_surf)
            )
            ts_time = ts_time - 2000

            base_t_idx = np.where((ts_time >= -2000) & (ts_time <= -1000))[0]
            m_base = np.mean(np.abs(all_layer_ts[:, base_t_idx]), axis=-1)
            t_idx = np.where((ts_time >= -75) & (ts_time <= -25))[0]
            signal_mag = np.mean(np.abs(all_layer_ts[:, t_idx]), axis=-1) - m_base

            # Compute anatomical predictors
            thickness = get_cortical_thickness(multilayer_mesh, n_layers)
            gainmat_fname = os.path.join(out_dir, f'SPMgainmatrix_{data_base}_1.mat')
            lf_rmse = get_lead_field_rmse(gainmat_fname, n_layers, verts_per_surf)
            scalp_mesh_fname = os.path.join(preproc_data_dir, subject, 't1wscalp_2562.surf.gii')
            orientations = get_orientation(scalp_mesh_fname, multilayer_mesh, ds_pial, verts_per_surf)
            distances = get_dist_to_scalp(scalp_mesh_fname, ds_pial)

            # Z-score each anatomical predictor (note: invert distance to scalp)
            z_thickness = zscore(thickness)
            z_lf_rmse = zscore(lf_rmse)
            z_orient = zscore(orientations)
            z_inv_dist = zscore(-distances)

            # Composite anatomical score
            anatomical_score = z_thickness + z_lf_rmse + z_orient + z_inv_dist

            # Restrict to roi_idx
            roi_anat_score = anatomical_score[roi_idx]
            roi_signal = signal_mag[roi_idx]

            # Select top 1% signal vertices within ROI
            signal_threshold = np.percentile(roi_signal, 99)
            high_signal_mask = roi_signal >= signal_threshold
            candidate_indices = roi_idx[high_signal_mask]

            # Among them, select the vertex with the best anatomical suitability
            candidate_anat_scores = anatomical_score[candidate_indices]
            best_idx_local = np.argmax(candidate_anat_scores)
            prior = candidate_indices[best_idx_local]

            # Get source time series from middle surface layer
            prior_ts = all_layer_ts[prior, :]

            # Run sliding time window model comparison
            [Fs, wois] = sliding_window_model_comparison(
                prior,
                nas,
                lpa,
                rpa,
                mri_fname,
                layer_fnames,
                base_fname,
                spm_instance=spm,
                viz=False,
                invert_kwargs={
                    'patch_size': patch_size,
                    'n_temp_modes': 4,
                    'n_spatial_modes': rank['mag'],
                    'win_size': 25,
                    'win_overlap': True
                }
            )

            # Save results
            np.savez(
                out_fname,
                subject=subject,
                session=session,
                epoch='motor_epoch',
                prior=prior,
                prior_ts=prior_ts,
                ts_time=ts_time,
                Fs=Fs,
                wois=wois-2000
            )


if __name__ == '__main__':
    spm = spm_standalone.initialize()

    #for patch_size in [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10]:
    for patch_size in [5.0]:
        run(patch_size, spm)

    spm.terminate()