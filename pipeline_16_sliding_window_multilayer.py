import json
import os
import os.path as op
import shutil
import sys

import numpy as np

from lameg.laminar import sliding_window_model_comparison
from lameg.util import get_surface_names, spm_context

from utilities import files
from utilities.utils import get_fiducial_coords

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

    # Native space MRI to use for coregistration
    mri_fname = os.path.join(
        sub_path,
        't1w.nii'
    )

    surf_dir = os.path.join(
        sub_path,
        'surf'
    )

    n_layers = 11

    # Get name of each mesh that makes up the layers of the multilayer mesh - these will be used for the source
    # reconstruction
    layer_fnames = get_surface_names(
        n_layers,
        surf_dir,
        'link_vector.fixed'
    )

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


    if os.path.exists(fname) and not os.path.exists(out_fname):
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

        base_fname = os.path.join(ses_out_path, f'{data_base}.mat')

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
                    patch_size=patch_size,
                    n_temp_modes=sliding_n_temp_modes,
                    win_size=win_size,
                    win_overlap=win_overlap,
                    spm_instance=spm,
                    viz=False
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


    run(index, session_index, epoch_type, epoch, json_file)

