import glob
import json
import os
import os.path as op
import shutil
import sys

import numpy as np

from lameg.laminar import sliding_window_model_comparison
from lameg.util import spm_context

from utilities import files
from utilities.utils import get_fiducial_coords, get_subject_sessions_idx


def run(subj_idx, ses_idx, epo_type, epo, json_file):
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

    # Patch size to use for inversion
    patch_size = 5
    # Number of temporal models for sliding time window inversion
    sliding_n_temp_modes = 4
    # Size of sliding window (in ms)
    win_size = 50
    # Whether or not windows overlap
    win_overlap = True

    tmp_dir = os.path.join(out_path, f'mlayer_{subj_idx}_{ses_idx}_{epo_type}_{epo}')
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

    n_layers = 11

    # Get name of each mesh that makes up the layers of the multilayer mesh - these will be used for the source
    # reconstruction
    orientation_method = 'link_vector.fixed'
    layers = np.linspace(1, 0, n_layers)
    layer_fnames = []
    for layer in layers:

        if layer == 1:
            name = f'pial.ds.{orientation_method}'
        elif layer == 0:
            name = f'white.ds.{orientation_method}'
        else:
            name = f'{layer:.3f}.ds.{orientation_method}'

        shutil.copy(os.path.join(surf_dir, f'{name}.gii'),
                    os.path.join(tmp_dir, f'{name}.gii'))
        layer_fnames.append(os.path.join(tmp_dir, f'{name}.gii'))
        shutil.copy(os.path.join(surf_dir, f'FWHM5.00_{name}.mat'),
                    os.path.join(tmp_dir, f'FWHM5.00_{name}.mat'))

    ses_out_path = os.path.join(
        out_path,
        subject_id,
        session_id
    )
    out_fname = os.path.join(ses_out_path, f'multilaminar_results_{epo_type}-epo.npz')
    data_file = os.path.join(ses_path,
        f'spm/pmcspm_converted_autoreject-{subject_id}-{session_id}-{epo_type}-epo.mat'
    )
    fname = os.path.join(ses_out_path, f'localizer_results_{epo_type}-epo.npz')
    if epo_type=='visual':
        out_fname = os.path.join(ses_out_path, f'multilaminar_results_{epo}_{epo_type}-epo.npz')
        data_file = os.path.join(ses_path,
            f'spm/{epo}_pmcspm_converted_autoreject-{subject_id}-{session_id}-{epo_type}-epo.mat'
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
        with np.load(fname) as data:
            cluster_vtx = data['cluster_vtx']

        cluster_layer_fs = []
        with spm_context() as spm:
            for c_idx in range(len(cluster_vtx)):
                subj_vtx = cluster_vtx[c_idx]

                # Run sliding time window model comparison
                [Fs, wois] = sliding_window_model_comparison(
                    subj_vtx,
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
                        'n_temp_modes': sliding_n_temp_modes,
                        'win_size': win_size,
                        'win_overlap': win_overlap
                    }
                )

                woi_time = np.array([np.mean(x) for x in wois])
                cluster_layer_fs.append(Fs)
        cluster_layer_fs = np.array(cluster_layer_fs)
        np.savez(
            out_fname,
            cluster_vtx=cluster_vtx,
            cluster_layer_fs=cluster_layer_fs,
            fe_time=woi_time
        )
    shutil.rmtree(tmp_dir)

if __name__=='__main__':

    subjects, sessions = get_subject_sessions_idx()
    epoch_types = ['visual','visual','motor']
    epochs = ['rdk','instr','']

    all_subjects=[]
    all_sessions=[]
    all_epoch_types=[]
    all_epochs=[]

    for epoch_type, epoch in zip(epoch_types, epochs):
        for subject, session in zip(subjects, sessions):
            all_subjects.append(subject)
            all_sessions.append(session)
            all_epoch_types.append(epoch_type)
            all_epochs.append(epoch)

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


    run(all_subjects[index], all_sessions[index], all_epoch_types[index], all_epochs[index], json_file)

