import sys
import json
import os.path as op

from mne import read_epochs

from extra.tools import dump_the_dict
from utilities import files
from os import sep
import nibabel as nb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_surface_plotting import plot_surf

def run(index, json_file):
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

        epo_paths = files.get_files(session, subject_id + "-" + session_id + "-001", "-epo.fif")[2]
        epo_types = []
        for epo in epo_paths:
            epo_types.append(epo.split(sep)[-1].split("-")[5])

        for epo_type in epo_types:
            res_fnames = files.get_files(op.join(session,'spm'), "invert_mspm_converted", epo_type+"-epo_localizer_results.json")[2]
            res_fnames.sort()

            epo_paths = files.get_files(session, "autoreject", epo_type+"-epo.fif")[2]

            fig = plt.figure(figsize=(18, 18))

            for run_idx,res_fname in enumerate(res_fnames):
                numero = res_fname.split(sep)[-1].split("-")[5]

                epo_path = [i for i in epo_paths if numero + '-' + epo_type + '-epo' in i][0]

                epochs = read_epochs(epo_path, verbose=False, preload=True)
                epochs.pick_types(meg=True, ref_meg=False)
                epo_data=epochs.get_data()

                with open(res_fname) as results_file:
                    results = json.load(results_file)

                pial_idx = results['surfaces'].index('pial')
                pial_surf_fname = results['surf_fname'][pial_idx]
                inflated_pial_surf_fname=op.join(sep.join(pial_surf_fname.split(sep)[0:-1]),'pial.ds.inflated.nodeep.gii')
                pial_res_fname = results['res_fname'][pial_idx]
                pial_surf = nb.load(inflated_pial_surf_fname)
                pial_vertices, pial_faces = pial_surf.darrays[0].data, pial_surf.darrays[1].data
                pial_res = pd.read_csv(pial_res_fname, sep="\t",header=None).values

                white_idx = results['surfaces'].index('white')
                white_res_fname = results['res_fname'][white_idx]
                white_res = pd.read_csv(white_res_fname, sep="\t", header=None).values

                mean_res=0.5*(pial_res+white_res)
                thresh=np.percentile(mean_res,99.5)
                peak_idx=np.argmax(mean_res)

                pial_mu_fname = results['mu_fname'][pial_idx]
                pialMU = pd.read_csv(pial_mu_fname, sep="\t", header=None).values

                white_mu_fname = results['mu_fname'][white_idx]
                whiteMU = pd.read_csv(white_mu_fname, sep="\t", header=None).values

                pial_it_fname = results['it_fname'][pial_idx]
                pialIt = pd.read_csv(pial_it_fname, sep="\t", header=None).values-1

                white_it_fname = results['it_fname'][white_idx]
                whiteIt = pd.read_csv(white_it_fname, sep="\t", header=None).values-1

                all_pial_trials=np.zeros((len(epochs), epo_data.shape[2]))
                all_white_trials = np.zeros((len(epochs), epo_data.shape[2]))
                for t in range(len(epochs)):
                    all_pial_trials[t,:] = np.matmul(pialMU[peak_idx,:], np.squeeze(epo_data[t,:,:]))
                    all_white_trials[t, :] = np.matmul(whiteMU[peak_idx, :], np.squeeze(epo_data[t, :, :]))

                int(peak_idx)
                results['times']=epochs.times.tolist()
                results['pial_source']=np.mean(all_pial_trials, axis=0).tolist()
                results['white_source'] = np.mean(all_white_trials, axis=0).tolist()
                dump_the_dict(res_fname, results)

                rotate=[90]
                x_rotate = 270
                if 'motor' in epo_type:
                    rotate=[0]
                    x_rotate=0
                elif 'visual' in epo_type:
                    rotate = [0]
                    x_rotate = -90

                ax = plt.subplot(len(res_fnames), 3, run_idx*3+1, xlim=[-.98, +.98], ylim=[-.98, +.98],
                                 aspect=1, frameon=False,
                                 xticks=[], yticks=[])
                plot_surf(vertices=pial_vertices, faces=pial_faces, overlay=mean_res, rotate=rotate, x_rotate=x_rotate,
                          ax=ax, colorbar=False, alpha_colour=(mean_res > thresh).astype(int))
                ax = plt.subplot(len(res_fnames), 3, run_idx*3+2, xlim=[-.98, +.98], ylim=[-.98, +.98],
                                 aspect=1, frameon=False,
                                 xticks=[], yticks=[])
                overlay = np.zeros(mean_res.shape)
                overlay[peak_idx] = 10
                plot_surf(vertices=pial_vertices, faces=pial_faces, overlay=overlay, rotate=rotate, x_rotate=x_rotate,
                          ax=ax, colorbar=False, cmap='jet', alpha_colour=(overlay > 0).astype(int), vmin=0, vmax=4, transparency=.25)
                plt.subplot(len(res_fnames), 3, run_idx*3+3)
                t_g0 = np.where(epochs.times > 0)[0]
                pial_mean = np.mean(all_pial_trials, axis=0)
                max_idx = np.argmax(np.abs(pial_mean[t_g0]))
                if pial_mean[t_g0][max_idx] > 0:
                    pial_mean = -1 * pial_mean
                white_mean = np.mean(all_white_trials, axis=0)
                max_idx = np.argmax(np.abs(white_mean[t_g0]))
                if white_mean[t_g0][max_idx] > 0:
                    white_mean = -1 * white_mean
                plt.plot(epochs.times, pial_mean)
                plt.plot(epochs.times, white_mean)
                plt.legend(['pial', 'white'])
                plt.xlabel('Time (s)')
                plt.ylabel('Source density (pAm/mm^2)')

            fig.suptitle("{}-{}-{}".format(subject_id, session_id,epo_type))
            plt.savefig(
                op.join(
                    session,
                    'spm',
                    'invert_mspm_converted_autoreject-{}-{}-{}-epo_localizer_results.png'.format(
                        subject_id, session_id, epo_type)
                ),
                dpi=300,
                pad_inches=0,
                transparent=True
            )
            plt.close("all")




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