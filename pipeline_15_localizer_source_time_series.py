import json
import os
import os.path as op
import shutil
import sys

import numpy as np
import nibabel as nib
from lameg.invert import coregister, invert_ebb, invert_msp, load_source_time_series
from lameg.util import matlab_context, make_directory

from utilities import files
from utilities.utils import get_fiducial_coords, get_roi_idx, find_clusters


def run(subj_idx, ses_idx, epo_type, epo, localizer_woi, roi_hemi, roi_regions, json_file, thresh=99.99):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]
    out_path = parameters["output_path"]
    der_path = op.join(path, "derivatives")
    proc_path = op.join(der_path, "processed")

    subjects = files.get_folders(proc_path, 'sub-', '')[2]
    subjects.sort()
    subject = subjects[subj_idx]
    subject_id = subject.split("/")[-1]
    print("ID:", subject_id)

    sub_path = op.join(proc_path, subject_id)

    sessions = files.get_folders(subject, 'ses', '')[2]
    sessions.sort()
    session = sessions[ses_idx]
    session_id = session.split("/")[-1]

    ses_path = op.join(sub_path, session_id)

    lpa, nas, rpa = get_fiducial_coords(subject_id, json_file)

    data_file = os.path.join(
        ses_path,
        f'spm/pmcspm_converted_autoreject-{subject_id}-{session_id}-{epo_type}-epo.mat'
    )
    if len(epo):
        data_file = os.path.join(
            ses_path,
            f'spm/{epo}_pmcspm_converted_autoreject-{subject_id}-{session_id}-{epo_type}-epo.mat'
        )

    # Native space MRI to use for coregistration
    mri_fname = os.path.join(
        sub_path,
        't1w.nii'
    )

    surf_dir = os.path.join(
        sub_path,
        'surf'
    )
    multilayer_mesh_fname = os.path.join(surf_dir, 'multilayer.11.ds.link_vector.fixed.gii')

    # Load multilayer mesh and compute the number of vertices per layer
    mesh = nib.load(multilayer_mesh_fname)
    n_layers = 11
    verts_per_surf = int(mesh.darrays[0].data.shape[0] / n_layers)

    # Get name of each mesh that makes up the layers of the multilayer mesh - these will be used for the source
    # reconstruction
    ds_pial = nib.load(os.path.join(surf_dir, 'pial.ds.gii'))
    ds_mid = nib.load(os.path.join(surf_dir, '0.300.ds.link_vector.fixed.gii'))

    # Extract base name and path of data file
    data_path, data_file_name = os.path.split(data_file)
    data_base = os.path.splitext(data_file_name)[0]

    _ = make_directory(
        out_path,
        [subject_id, session_id],
        check=False
    )
    ses_out_path = os.path.join(
        out_path,
        subject_id,
        session_id
    )

    shutil.copy(
        os.path.join(data_path, f'{data_base}.mat'),
        os.path.join(ses_out_path, f'{data_base}.mat')
    )
    shutil.copy(
        os.path.join(data_path, f'{data_base}.dat'),
        os.path.join(ses_out_path, f'{data_base}.dat')
    )

    base_fname = os.path.join(ses_out_path, f'{data_base}.mat')

    # Patch size to use for inversion
    patch_size = 5
    # Number of temporal modes to use for EBB inversion
    n_temp_modes = 4

    with matlab_context() as eng:
        # Coregister data to multilayer mesh
        coregister(
            nas,
            lpa,
            rpa,
            mri_fname,
            multilayer_mesh_fname,
            base_fname,
            mat_eng=eng,
            viz=False
        )

        [_, _, MU] = invert_ebb(
            multilayer_mesh_fname,
            base_fname,
            n_layers,
            woi=localizer_woi,
            patch_size=patch_size,
            n_temp_modes=n_temp_modes,
            return_mu_matrix=True,
            mat_eng=eng,
            viz=False
        )

        layer_vertices = np.arange(verts_per_surf)
        all_layer_ts, time = load_source_time_series(
            base_fname,
            mu_matrix=MU,
            vertices=layer_vertices,
            mat_eng=eng
        )

    roi_idx = get_roi_idx(subject_id, surf_dir, roi_hemi, roi_regions, ds_pial)

    mask = np.zeros(verts_per_surf)
    mask[roi_idx] = 1

    m_layer_max = np.max(np.abs(all_layer_ts), axis=-1) * mask

    cluster_thresh = np.percentile(m_layer_max, thresh)

    cluster_mask = np.where(m_layer_max >= cluster_thresh)[0]
    clusters = find_clusters(mesh.darrays[1].data, cluster_mask)

    cluster_vtx = []
    cluster_coord = []
    cluster_ts = []

    for c_idx in range(len(clusters)):
        cluster = clusters[c_idx]

        norm_cluster_max = m_layer_max[cluster] / np.max(m_layer_max[cluster])
        max_c_idx = np.argmax(norm_cluster_max)
        max_v_idx = cluster[max_c_idx]

        max_coord = ds_mid.darrays[0].data[max_v_idx, :]

        with matlab_context() as eng:
            # Coregister data to multilayer mesh
            coregister(
                nas,
                lpa,
                rpa,
                mri_fname,
                multilayer_mesh_fname,
                base_fname,
                mat_eng=eng,
                viz=False
            )

            [_, _, MU] = invert_msp(
                multilayer_mesh_fname,
                base_fname,
                n_layers,
                priors=[max_v_idx],
                patch_size=patch_size,
                n_temp_modes=n_temp_modes,
                return_mu_matrix=True,
                mat_eng=eng,
                viz=False
            )

            prior_layer_ts, time = load_source_time_series(
                base_fname,
                mu_matrix=MU,
                vertices=[max_v_idx],
                mat_eng=eng
            )

        cluster_vtx.append(max_v_idx)
        cluster_coord.append(max_coord)
        cluster_ts.append(prior_layer_ts)

    cluster_vtx = np.array(cluster_vtx)
    cluster_coord = np.array(cluster_coord)
    cluster_ts = np.array(cluster_ts)

    out_fname = os.path.join(ses_out_path, f'localizer_results_{epo_type}-epo.npz')
    if len(epo):
        out_fname = os.path.join(ses_out_path, f'localizer_results_{epo}_{epo_type}-epo.npz')
    np.savez(
        out_fname,
        cluster_vtx=cluster_vtx,
        cluster_coord=cluster_coord,
        cluster_ts=cluster_ts,
        ts_time=time * 1000,
    )


if __name__=='__main__':
    # parsing command line arguments
    try:
        index = int(sys.argv[1])
    except:
        print("incorrect subject index")
        sys.exit()

    try:
        session_index = int(sys.argv[2])
    except:
        print("incorrect session index")
        sys.exit()

    try:
        epoch_type = sys.argv[3]
    except:
        print("incorrect epoch type")
        sys.exit()

    try:
        epoch = sys.argv[4]
    except:
        print("incorrect epoch")
        sys.exit()

    try:
        json_file = sys.argv[5]
        print("USING:", json_file)
    except:
        json_file = "settings.json"
        print("USING:", json_file)

    if epoch_type== 'motor':
        localizer_woi=[-125,-25]
        roi_hemi='lh'
        roi_regions=['precentral', 'postcentral']
        thresh = 99.95
    else:
        if epoch== 'instr':
            localizer_woi=[2550, 2650]
        else:
            localizer_woi=[50, 150]
        roi_hemi=None
        roi_regions=['pericalcarine']
        thresh = 99.99

    run(index, session_index, epoch_type, epoch, localizer_woi, roi_hemi, roi_regions, json_file, thresh=thresh)