import json
import os
import os.path as op
import shutil
import sys

import numpy as np

from lameg.invert import coregister, invert_msp, load_source_time_series
from lameg.util import matlab_context

from utilities import files
from utilities.utils import get_fiducial_coords


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
    if len(epo):
        out_fname = os.path.join(ses_out_path, f'localizer_results_{epo}_{epo_type}-epo_{condition}.npz')
        data_file = os.path.join(
            ses_path,
            f'spm/{epo}_pm{condition}_cspm_converted_autoreject-{subject_id}-{session_id}-{epo_type}-epo.mat'
        )
        fname = os.path.join(ses_out_path, f'localizer_results_{epo}_{epo_type}-epo.npz')


    if os.path.exists(fname) and not os.path.exists(out_fname):
        data=np.load(fname)
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

        cluster_ts = []
        for c_idx in range(len(cluster_vtx)):
            subj_vtx = cluster_vtx[c_idx]

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

                [_, _, MU] = invert_msp(
                    multilayer_mesh_fname,
                    base_fname,
                    n_layers,
                    priors=[subj_vtx],
                    patch_size=patch_size,
                    n_temp_modes=n_temp_modes,
                    return_mu_matrix=True,
                    mat_eng=eng,
                    viz=False
                )

                prior_layer_ts, time = load_source_time_series(
                    base_fname,
                    mu_matrix=MU,
                    vertices=[subj_vtx],
                    mat_eng=eng
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
        condition = sys.argv[5]
    except:
        print("incorrect condition")
        sys.exit()

    try:
        json_file = sys.argv[6]
        print("USING:", json_file)
    except:
        json_file = "settings.json"
        print("USING:", json_file)


    run(index, session_index, epoch_type, epoch, condition, json_file)
