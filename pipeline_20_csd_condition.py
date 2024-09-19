import glob
import json
import os
import os.path as op
import shutil
import sys

import numpy as np
import nibabel as nib

from lameg.invert import invert_ebb, coregister, load_source_time_series
from lameg.laminar import compute_csd
from lameg.util import spm_context

from utilities import files
from utilities.utils import get_fiducial_coords, get_subject_sessions_idx


def run(subj_idx, ses_idx, epo_type, epo, conditions, json_file, spm_instance=None):

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

    # Sampling rate
    s_rate = 600

    # Native space MRI to use for coregistration
    mri_fname = os.path.join(
        sub_path,
        't1w.nii'
    )

    surf_dir = os.path.join(
        sub_path,
        'surf'
    )

    multilayer_mesh_fname = os.path.join(surf_dir, 'multilayer.15.ds.link_vector.fixed.gii')

    # Load multilayer mesh and compute the number of vertices per layer
    mesh = nib.load(multilayer_mesh_fname)
    n_layers = 15
    verts_per_surf = int(mesh.darrays[0].data.shape[0] / n_layers)

    ses_out_path = os.path.join(
        out_path,
        subject_id,
        session_id
    )
    data_file = os.path.join(
        ses_path,
        f'spm/pmcspm_converted_autoreject-{subject_id}-{session_id}-{epo_type}-epo.mat'
    )

    fname = os.path.join(ses_out_path, f'localizer_results_{epo_type}-epo.npz')
    if epo_type == 'visual':
        data_file = os.path.join(
            ses_path,
            f'spm/{epo}_pmcspm_converted_autoreject-{subject_id}-{session_id}-{epo_type}-epo.mat'
        )
        fname = os.path.join(ses_out_path, f'localizer_results_{epo}_{epo_type}-epo.npz')

    if os.path.exists(fname):# and not os.path.exists(out_fname):
        with np.load(fname) as data:
            cluster_vtx = data['cluster_vtx']

        # Extract base name and path of data file
        data_path, data_file_name = os.path.split(data_file)
        data_base = os.path.splitext(data_file_name)[0]

        shutil.copy(
            os.path.join(data_path, f'{data_base}.mat'),
            os.path.join(ses_out_path, f'{data_base}.mat')
        )
        shutil.copy(
            os.path.join(data_path, f'{data_base}.dat'),
            os.path.join(ses_out_path, f'{data_base}.dat')
        )

        base_data_fname = os.path.join(ses_out_path, f'{data_base}.mat')

        # Patch size to use for inversion (in this case it matches the simulated patch size)
        patch_size = 5
        # Number of temporal modes to use for EBB inversion
        n_temp_modes = 4

        with spm_context(spm=spm_instance) as spm:
            # Coregister data to multilayer mesh
            coregister(
                nas,
                lpa,
                rpa,
                mri_fname,
                multilayer_mesh_fname,
                base_data_fname,
                spm_instance=spm
            )

            [_, _, MU] = invert_ebb(
                multilayer_mesh_fname,
                base_data_fname,
                n_layers,
                patch_size=patch_size,
                n_temp_modes=n_temp_modes,
                return_mu_matrix=True,
                spm_instance=spm
            )

            for condition in conditions:
                out_fname = os.path.join(ses_out_path, f'csd_results_{epo_type}-epo_{condition}.npz')
                cond_file = os.path.join(
                    ses_path,
                    f'spm/pm{condition}_cspm_converted_autoreject-{subject_id}-{session_id}-{epo_type}-epo.mat'
                )
                if epo_type == 'visual':
                    out_fname = os.path.join(ses_out_path, f'csd_results_{epo}_{epo_type}-epo_{condition}.npz')
                    cond_file = os.path.join(
                        ses_path,
                        f'spm/{epo}_pm{condition}_cspm_converted_autoreject-{subject_id}-{session_id}-{epo_type}-epo.mat'
                    )

                cluster_layer_ts=[]
                cluster_csd = []
                cluster_smooth_csd = []
                for c_idx in range(len(cluster_vtx)):
                    subj_vtx = cluster_vtx[c_idx]

                    layer_verts = [l * int(verts_per_surf) + subj_vtx for l in range(n_layers)]
                    layer_coords = mesh.darrays[0].data[layer_verts, :]
                    layer_dists = np.sqrt(np.sum(np.diff(layer_coords, axis=0) ** 2, axis=1))

                    # Get source time series for each layer
                    layer_ts, time, _ = load_source_time_series(cond_file, vertices=layer_verts, mu_matrix=MU)

                    # Compute CSD and smoothed CSD
                    [csd, smooth_csd] = compute_csd(layer_ts, np.mean(layer_dists), s_rate, smoothing='cubic')

                    cluster_layer_ts.append(layer_ts)
                    cluster_csd.append(csd)
                    cluster_smooth_csd.append(smooth_csd)

                cluster_layer_ts = np.array(cluster_layer_ts)
                cluster_csd = np.array(cluster_csd)
                cluster_smooth_csd = np.array(cluster_smooth_csd)

                np.savez(
                    out_fname,
                    cluster_vtx=cluster_vtx,
                    cluster_layer_ts=cluster_layer_ts,
                    cluster_csd=cluster_csd,
                    cluster_smooth_csd=cluster_smooth_csd,
                    time=time
                )

if __name__=='__main__':

    subjects, sessions = get_subject_sessions_idx()
    epo_types=['visual','visual','motor']
    epos=['rdk','instr','']
    conditions = ['congruent', 'incongruent', 'coherence-low', 'coherence-med', 'coherence-high',
                  'congruent_coherence-low', 'congruent_coherence-med', 'congruent_coherence-high',
                  'incongruent_coherence-low', 'incongruent_coherence-med', 'incongruent_coherence-high']

    with spm_context() as spm:
        base_data_dir = '/home/bonaiuto/laminar_erf/output/'

        for epo_type,epo in zip(epo_types,epos):
            for subject_idx, session_idx in zip(subjects, sessions):
                run(subject_idx, session_idx, epo_type, epo, conditions, 'settings.json', spm_instance=spm)

