import json
import os.path as op
import sys
from os import sep
from utilities import files
import nibabel as nb
import pandas as pd
import numpy as np
from mne import read_epochs, pick_types
from scipy.interpolate import interp2d, interp1d
from scipy.signal import savgol_filter


def compute_csd(surf_tcs, mean_dist, n_surfs, smooth=True):
    # Compute CSD
    nd = 1
    spacing = mean_dist * 10 ** -3

    csd = np.zeros((n_surfs, surf_tcs.shape[1]))
    for t in range(surf_tcs.shape[1]):
        phi = surf_tcs[:, t]
        csd[0, t] = surf_tcs[0, t]
        csd[1, t] = surf_tcs[1, t]
        for z in range(2, n_surfs - 3):
            csd[z, t] = (phi[z + 2] - 2 * phi[z] + phi[z - 2]) / ((nd * spacing) ** 2)
        csd[-2, t] = surf_tcs[-2, t]
        csd[-1, t] = surf_tcs[-1, t]

    csd_smooth=None
    if smooth:
        # interpolate CSD in space
        y = np.linspace(0, n_surfs - 1, n_surfs)
        Yi = np.linspace(0, n_surfs - 1, 500)

        f = interp1d(y, csd, kind='cubic', axis=0)
        csd_smooth = f(Yi)

        csd_smooth = savgol_filter(csd_smooth, 51, 3, axis=1)

    return (csd, csd_smooth)

def run(index, json_file):
    epo_types = ['motor','visual1','visual2']

    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]
    der_path = op.join(path, "derivatives")
    proc_path = op.join(der_path, "processed")

    subjects = files.get_folders_files(proc_path)[0]
    subjects.sort()

    subject = subjects[index]
    subject_id = subject.split("/")[-1]

    sessions = files.get_folders(subject, 'ses', '')[2]
    sessions.sort()

    n_surfs=11

    print("ID:", subject_id)

    for session in sessions:
        session_id = session.split("/")[-1]
        print("Session:", session_id)

        for epo_type in epo_types:
            print("Epoch:", epo_type)

            res_fnames = files.get_files(op.join(session, 'spm'), "invert_fmspm_converted_autoreject",
                                         epo_type + "-epo_multilayer_results.json")[2]
            res_fnames.sort()

            for res_fname in res_fnames:
                numero = res_fname.split(sep)[-1].split("-")[5]
                print("Run:", numero)

                with open(res_fname) as results_file:
                    results = json.load(results_file)

                # Load multilayer surface
                multilayer_surf = nb.load(results['surf_fname']);
                multilayer_vertices, multilayer_faces = multilayer_surf.darrays[0].data, multilayer_surf.darrays[1].data

                # Load pial surface
                pial_surf_fname = op.join(sep.join(results['surf_fname'].split(sep)[0:-1]), 'pial.ds.inflated.nodeep.gii')
                pial_surf = nb.load(pial_surf_fname);
                pial_vertices, pial_faces = pial_surf.darrays[0].data, pial_surf.darrays[1].data

                # Load inversion results and MU matrix
                MU = pd.read_csv(results['mu_fname'], sep="\t", header=None).values

                n_vertices_per_surf = int(multilayer_vertices.shape[0] / n_surfs)

                # Load data
                epo_path = op.join(session,
                                   'autoreject-' + subject_id + "-" + session_id + "-" + numero + "-" + epo_type + "-epo.fif")
                epochs = read_epochs(epo_path, verbose=False, preload=True)
                # epochs = epochs.filter(0,30)
                meg_chans = pick_types(epochs.info, meg=True, ref_meg=False)
                epo_data = epochs.get_data()
                epo_data = epo_data[:, meg_chans, :]
                # Average over trials
                evoked = np.mean(epo_data, axis=0)

                all_csd=np.zeros((n_vertices_per_surf,n_surfs,evoked.shape[1]))
                for peak_idx in range(n_vertices_per_surf):

                    # Check surface spacing
                    peak_coord = np.zeros((n_surfs, 3))
                    for i in range(n_surfs):
                        peak_coord[i, :] = multilayer_vertices[i * n_vertices_per_surf + peak_idx, :]
                    dists = np.sqrt(np.sum((peak_coord[1:, :] - peak_coord[0:-1, :]) ** 2, axis=1))
                    mean_dist = np.mean(dists)

                    # Get source time course in each surface
                    surf_tcs = np.zeros((n_surfs, evoked.shape[1]))
                    for i in range(n_surfs):
                        surf_peak_idx = i * n_vertices_per_surf + peak_idx
                        surf_tcs[n_surfs - i - 1, :] = np.matmul(MU[surf_peak_idx, :].reshape(1, -1), evoked)

                    (csd, csd_smooth) = compute_csd(surf_tcs, mean_dist, n_surfs, smooth=False)
                    all_csd[peak_idx,:,:]=csd

                fname=op.join(session,
                              'fmspm_converted_autoreject-{}-{}-{}-{}-epo-csd.npy'.format(subject_id,
                                                                                          session_id,
                                                                                          numero,
                                                                                          epo_type))
                np.save(fname, all_csd)


if __name__=='__main__':
    # parsing command line arguments
    try:
        index = int(sys.argv[1])
    except:
        print("incorrect arguments")
        sys.exit()

    try:
        json_file = sys.argv[2]
        print("USING:", json_file)
    except:
        json_file = "settings.json"
        print("USING:", json_file)

    run(index, json_file)