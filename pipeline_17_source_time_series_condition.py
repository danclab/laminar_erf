import glob
import json
import os
import os.path as op
import shutil
import sys

import numpy as np

from lameg.invert import coregister, invert_msp, load_source_time_series
from lameg.util import spm_context

from utilities import files
from utilities.utils import get_fiducial_coords, get_subject_sessions_idx


def run(subj_idx, ses_idx, epo_type, epo, condition, json_file):
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

    tmp_dir = os.path.join(out_path, f'ts_cond_{subj_idx}_{ses_idx}_{epo_type}_{epo}_{condition}')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    # Native space MRI to use for coregistration
    shutil.copy(os.path.join(sub_path, 't1w.nii'),
                os.path.join(tmp_dir, 't1w.nii'))
    surf_files = glob.glob(os.path.join(sub_path, 't1w*.gii'))
    for surf_file in surf_files:
        shutil.copy(surf_file, tmp_dir)
    mri_fname = os.path.join(tmp_dir, 't1w.nii')

    surf_dir = os.path.join(
        sub_path,
        'surf'
    )

    shutil.copy(os.path.join(surf_dir, 'multilayer.11.ds.link_vector.fixed.gii'),
                os.path.join(tmp_dir, 'multilayer.11.ds.link_vector.fixed.gii'))
    multilayer_mesh_fname = os.path.join(tmp_dir, 'multilayer.11.ds.link_vector.fixed.gii')
    shutil.copy(
        os.path.join(surf_dir, 'FWHM5.00_multilayer.11.ds.link_vector.fixed.mat'),
        os.path.join(tmp_dir, 'FWHM5.00_multilayer.11.ds.link_vector.fixed.mat'))

    n_layers = 11

    ses_out_path = os.path.join(
        out_path,
        subject_id,
        session_id
    )
    out_fname = os.path.join(ses_out_path, f'localizer_results_{epo_type}-epo_{condition}.npz')
    data_file = os.path.join(
        ses_path,
        f'spm/pm{condition}_cspm_converted_autoreject-{subject_id}-{session_id}-{epo_type}-epo.mat'
    )
    fname = os.path.join(ses_out_path, f'localizer_results_{epo_type}-epo.npz')
    if epo_type=='visual':
        out_fname = os.path.join(ses_out_path, f'localizer_results_{epo}_{epo_type}-epo_{condition}.npz')
        data_file = os.path.join(
            ses_path,
            f'spm/{epo}_pm{condition}_cspm_converted_autoreject-{subject_id}-{session_id}-{epo_type}-epo.mat'
        )
        fname = os.path.join(ses_out_path, f'localizer_results_{epo}_{epo_type}-epo.npz')
    data_path, data_file_name = os.path.split(data_file)
    data_base = os.path.splitext(data_file_name)[0]

    # Copy data files to tmp directory
    shutil.copy(
        os.path.join(data_path, f'{data_base}.mat'),
        os.path.join(tmp_dir, f'{data_base}.mat')
    )
    shutil.copy(
        os.path.join(data_path, f'{data_base}.dat'),
        os.path.join(tmp_dir, f'{data_base}.dat')
    )

    # Construct base file name for simulations
    base_fname = os.path.join(tmp_dir, f'{data_base}.mat')

    if os.path.exists(fname) and not os.path.exists(out_fname):
        data=np.load(fname)
        cluster_vtx = data['cluster_vtx']

        cluster_ts = []
        with spm_context() as spm:
            for c_idx in range(len(cluster_vtx)):
                subj_vtx = cluster_vtx[c_idx]

                # Patch size to use for inversion
                patch_size = 5
                # Number of temporal modes to use for EBB inversion
                n_temp_modes = 4

                # Coregister data to multilayer mesh
                coregister(
                    nas,
                    lpa,
                    rpa,
                    mri_fname,
                    multilayer_mesh_fname,
                    base_fname,
                    spm_instance=spm,
                    viz=False
                )

                [_, _, MU] = invert_msp(
                    multilayer_mesh_fname,
                    base_fname,
                    n_layers,
                    priors=[subj_vtx],
                    patch_size=patch_size,
                    n_temp_modes=n_temp_modes,
                    return_mu_matrix=True,
                    spm_instance=spm,
                    viz=False
                )

                prior_layer_ts, time, _ = load_source_time_series(
                    base_fname,
                    mu_matrix=MU,
                    vertices=[subj_vtx]
                )
                cluster_ts.append(prior_layer_ts)

        cluster_ts = np.array(cluster_ts)

        np.savez(
            out_fname,
            cluster_vtx = cluster_vtx,
            cluster_coord = data['cluster_coord'],
            cluster_ts = cluster_ts,
            ts_time = data['ts_time']
        )
    shutil.rmtree(tmp_dir)


if __name__=='__main__':

    subjects, sessions = get_subject_sessions_idx()
    epoch_types = ['visual', 'visual', 'motor']
    epochs = ['rdk', 'instr', '']
    conditions = ['congruent','incongruent','coherence-low','coherence-med','coherence-high',
                  'congruent_coherence-low','congruent_coherence-med','congruent_coherence-high',
                  'incongruent_coherence-low','incongruent_coherence-med','incongruent_coherence-high']

    all_subjects = []
    all_sessions = []
    all_epoch_types = []
    all_epochs = []
    all_conditions = []

    for epoch_type, epoch in zip(epoch_types, epochs):
        for subject, session in zip(subjects, sessions):
            for condition in conditions:
                all_subjects.append(subject)
                all_sessions.append(session)
                all_epoch_types.append(epoch_type)
                all_epochs.append(epoch)
                all_conditions.append(condition)

    # parsing command line arguments
    try:
        index = int(sys.argv[1])
    except:
        print("incorrect index")
        sys.exit()

    try:
        json_file = sys.argv[2]
        print("USING:", json_file)
    except:
        json_file = "settings.json"
        print("USING:", json_file)


    run(all_subjects[index], all_sessions[index], all_epoch_types[index], all_epochs[index], all_conditions[index], json_file)
