import sys
import json
from mne import read_epochs, read_events, set_log_level
from matplotlib import colors
import os.path as op
from os import sep
from utilities import files
import matplotlib.pylab as plt
from autoreject import compute_thresholds
import pandas as pd
import numpy as np

set_log_level(verbose=False)

epo_types = ['visual', 'motor']

def run(index, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]

    der_path = op.join(path, "derivatives")
    files.make_folder(der_path)
    proc_path = op.join(der_path, "processed")
    files.make_folder(proc_path)

    subjects = files.get_folders(proc_path,'sub-','')[2]
    subjects.sort()
    subject = subjects[index]
    subject_id = subject.split("/")[-1]

    print("ID:", subject_id)

    sub_path = op.join(proc_path, subject_id)
    files.make_folder(sub_path)

    sessions = files.get_folders(subject, 'ses', '')[2]
    sessions.sort()

    for session in sessions:
        session_id = session.split("/")[-1]

        sess_path = op.join(sub_path, session_id)
        files.make_folder(sess_path)

        qc_folder = op.join(sess_path, "QC")
        files.make_folder(qc_folder)

        cmap = colors.ListedColormap(["#FFFFFF", "#CFEEFA", "#FFDE00", "#FF9900", "#FF0000", "#000000"])
        boundaries = [-0.9, -0.1, 1.1, 10, 100, 1000, 10000]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

        for epo_type in epo_types:
            epo_path = op.join(sess_path, f"{subject_id}-{session_id}-{epo_type}-epo.fif")
            print("INPUT FILE:", epo_path)
            epochs = read_epochs(epo_path, verbose=False)

            beh_path = op.join(sess_path, f"{subject_id}-{session_id}-{epo_type}-beh.csv")
            print("INPUT BEHAV FILE:", beh_path)

            # Drop no responses
            beh = pd.read_csv(beh_path)
            rej_idx = (beh.response == 0 )
            if epo_type=='motor':
                rej_idx = rej_idx | (beh.correct == 0) | (beh.trial_congruence == 'incongruent')
            epochs = epochs.drop(np.where(rej_idx)[0])

            ch_thr = compute_thresholds(
                epochs,
                random_state=42,
                method="bayesian_optimization",
                verbose="progressbar",
                n_jobs=-1,
                augment=False
            )
            # save the thresholds
            ch_list = sorted(ch_thr.keys())
            data = epochs.get_data(picks=ch_list)
            thr = np.array([ch_thr[ch] for ch in ch_list], dtype=data.dtype)[None, :, None]
            counts = (data > thr).sum(axis=-1)  # (n_epochs, n_channels)
            results = counts.T.astype(np.int64)

            name = "{}-{}-{}".format(subject_id, session_id, epo_type)
            npy_path = op.join(qc_folder, name + ".npy")
            np.save(npy_path, results)
            img_path = op.join(qc_folder, name + "-epo-QC.png")
            print(results[:15, :15])
            print(np.min(results), np.max(results))
            print(np.unique(results))

            plt.rcParams.update({'font.size': 5})
            f, ax = plt.subplots(
                figsize=(20, 20),
                dpi=200
            )

            im = ax.imshow(
                results,
                aspect="auto",
                cmap=cmap,
                interpolation="none",
                norm=norm
            )
            f.colorbar(im, ax=ax, fraction=0.01, pad=0.01)
            ax.set_xlabel("Trials")
            ax.set_ylabel("Channels")
            ax.set_xticks(list(range(len(epochs))))
            ax.set_xticklabels([str(i) for i in range(1, len(epochs) + 1)])
            ax.set_yticks(list(range(len(ch_list))))
            ax.set_yticklabels(ch_list)
            ax.grid(color='w', linestyle='-', linewidth=0.2)
            ax.set_title(name)
            plt.savefig(
                img_path,
                bbox_inches="tight"
            )
            plt.close("all")


if __name__=='__main__':
    json_file = "settings.json"
    for index in range(8):
        run(index, json_file)