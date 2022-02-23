import sys
import json
import os.path as op

import mne
from mne import read_epochs

from extra.tools import dump_the_dict
from utilities import files
import matlab.engine
from os import sep
import nibabel as nb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_surface_plotting import plot_surf

def run(index, json_file, parasite):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]

    der_path = op.join(path, "derivatives")
    files.make_folder(der_path)
    proc_path = op.join(der_path, "processed")
    files.make_folder(proc_path)

    subjects = files.get_folders_files(proc_path)[0]
    subjects.sort()
    subject = subjects[index]
    subject_id = subject.split("/")[-1]
    print("ID:", subject_id)

    sessions = files.get_folders(subject, 'ses', '')[2]
    sessions.sort()

    for session in sessions:
        session_id = session.split("/")[-1]

        spm_paths = files.get_files(op.join(session,'spm'), "fmspm_converted_autoreject", "-epo.mat")[2]
        spm_paths.sort()
        epo_paths = files.get_files(session, "autoreject", "-epo.fif")[2]

        for spm_path in spm_paths:
            epo_type = spm_path.split(sep)[-1].split("-")[6]
            numero = spm_path.split(sep)[-1].split("-")[5]

            epo_path = [i for i in epo_paths if numero + '-' + epo_type + '-epo' in i][0]

            res_fname=parasite.invert_localize(path, subject_id, session_id, numero, epo_type, nargout=1)

            epochs = read_epochs(epo_path, verbose=False, preload=True)
            epochs.pick_types(meg=True, ref_meg=False)

            with open(res_fname) as results_file:
                results = json.load(results_file)

            pial_idx = results['surfaces'].index('pial')
            pial_mu_fname = results['mu_fname'][pial_idx]
            pialMU = pd.read_csv(pial_mu_fname, sep="\t", header=None).values
            pial_res_fname = results['res_fname'][pial_idx]
            pial_res = pd.read_csv(pial_res_fname, sep="\t", header=None).values

            white_idx = results['surfaces'].index('white')
            white_mu_fname = results['mu_fname'][white_idx]
            whiteMU = pd.read_csv(white_mu_fname, sep="\t", header=None).values
            white_res_fname = results['res_fname'][white_idx]
            white_res = pd.read_csv(white_res_fname, sep="\t", header=None).values

            mean_res = 0.5 * (pial_res + white_res)
            mean_peak_idx = np.argmax(mean_res)

            epo_data = epochs.get_data()
            pial_trials = np.zeros((len(epochs), epo_data.shape[2]))
            white_trials = np.zeros((len(epochs), epo_data.shape[2]))
            for t in range(len(epochs)):
                pial_trials[t, :] = np.matmul(pialMU[mean_peak_idx, :], np.squeeze(epo_data[t, :, :]))
                white_trials[t, :] = np.matmul(whiteMU[mean_peak_idx, :], np.squeeze(epo_data[t, :, :]))

            results['times'] = epochs.times.tolist()
            results['peak_idx'] = int(mean_peak_idx)
            results['pial_source'] = np.mean(pial_trials, axis=0).tolist()
            results['white_source'] = np.mean(white_trials, axis=0).tolist()
            dump_the_dict(res_fname, results)

            fig=plot_localizer_results(subject_id, session_id, numero, epo_type, results)
            [base, ext] = op.splitext(res_fname)
            plt.savefig(
                '{}.png'.format(base),
                dpi=300,
                pad_inches=0,
                transparent=False
            )
            plt.close("all")


def plot_localizer_results(subject_id, session_id, numero, epo_type, results):
    pial_idx = results['surfaces'].index('pial')
    pial_surf_fname = results['surf_fname'][pial_idx]
    inflated_pial_surf_fname = op.join(sep.join(pial_surf_fname.split(sep)[0:-1]), 'pial.ds.inflated.nodeep.gii')
    pial_res_fname = results['res_fname'][pial_idx]
    pial_surf = nb.load(inflated_pial_surf_fname)
    pial_vertices, pial_faces = pial_surf.darrays[0].data, pial_surf.darrays[1].data
    pial_res = pd.read_csv(pial_res_fname, sep="\t", header=None).values

    white_idx = results['surfaces'].index('white')
    white_res_fname = results['res_fname'][white_idx]
    white_res = pd.read_csv(white_res_fname, sep="\t", header=None).values
    mean_res = 0.5 * (pial_res + white_res)
    thresh = np.percentile(mean_res, 99.5)

    times=np.array(results['times'])

    rotate = [90]
    x_rotate = 270
    if 'motor' in epo_type:
        rotate = [0]
        x_rotate = 0
    elif 'visual' in epo_type:
        rotate = [0]
        x_rotate = -90
    fig = plt.figure(figsize=(18, 6))
    ax = plt.subplot(1, 3, 1, xlim=[-.98, +.98], ylim=[-.98, +.98],
                     aspect=1, frameon=False,
                     xticks=[], yticks=[])
    plot_surf(vertices=pial_vertices, faces=pial_faces, overlay=mean_res, rotate=rotate, x_rotate=x_rotate,
              ax=ax, colorbar=False, alpha_colour=(mean_res > thresh).astype(int))
    ax = plt.subplot(1, 3, 2, xlim=[-.98, +.98], ylim=[-.98, +.98],
                     aspect=1, frameon=False,
                     xticks=[], yticks=[])
    overlay = np.zeros(mean_res.shape)
    overlay[results['peak_idx']] = 10
    coords=np.array([pial_vertices[results['peak_idx'],:]])
    plot_surf(vertices=pial_vertices, faces=pial_faces, overlay=overlay, rotate=rotate, x_rotate=x_rotate,
              ax=ax, colorbar=False, cmap='jet', alpha_colour=(overlay > 0).astype(int), vmin=0, vmax=4,
              transparency=.25, coords=coords)
    plt.title('vtx={}'.format(results['peak_idx']))
    plt.subplot(1, 3, 3)
    t_g0 = np.where(times > 0)[0]
    pial_mean = np.array(results['pial_source'])
    white_mean = np.array(results['white_source'])
    max_idx = np.argmax(np.abs(pial_mean[t_g0]))
    if pial_mean[t_g0][max_idx] > 0:
        pial_mean = -1 * pial_mean
    max_idx = np.argmax(np.abs(white_mean[t_g0]))
    if white_mean[t_g0][max_idx] > 0:
        white_mean = -1 * white_mean
    plt.plot(times, pial_mean)
    plt.plot(times, white_mean)
    plt.legend(['pial', 'white'])
    plt.xlabel('Time (s)')
    plt.ylabel('Source density (pAm/mm^2)')

    fig.suptitle("{}-{}-{}".format(subject_id, session_id, numero))
    return fig

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

    parasite = matlab.engine.start_matlab()

    run(index, json_file, parasite)