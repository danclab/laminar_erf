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

# opening a json file
with open(json_file) as pipeline_file:
    parameters = json.load(pipeline_file)

path = parameters["dataset_path"]
sfreq = parameters["downsample_dataset"]
hi_pass = parameters["hi_pass_filter"]
sub_path = op.join(path, "daw")

der_path = op.join(path, "derivatives")
files.make_folder(der_path)
proc_path = op.join(der_path, "processed")
files.make_folder(proc_path)

subjects = files.get_folders_files(proc_path)[0]
subjects.sort()
subject = subjects[index]
subject_id = subject.split("/")[-1]

print("ID:", subject_id)

sub_path = op.join(proc_path, subject_id)
files.make_folder(sub_path)

sessions = files.get_folders(subject,'ses','')[2]
sessions.sort()

for session in sessions:
    session_id = session.split("/")[-1]

    sess_path = op.join(sub_path, session_id)
    files.make_folder(sess_path)

    qc_folder = op.join(sess_path, "QC")
    files.make_folder(qc_folder)

    epo_paths = files.get_files(sess_path, "sub", "-epo.fif")[2]
    epo_paths.sort()
    beh_paths = files.get_files(sess_path, "sub", "-beh.csv")[2]
    beh_paths.sort()
    eve_paths = files.get_files(sess_path, "sub", "-eve.fif")[2]
    eve_paths.sort()

    cmap = colors.ListedColormap(["#FFFFFF", "#CFEEFA", "#FFDE00", "#FF9900", "#FF0000", "#000000"])
    boundaries = [-0.9, -0.1, 1.1, 10, 100, 1000, 10000]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    # for epo in epo_paths:
    for epo in epo_paths:
        print("INPUT FILE:", epo)
        numero = epo.split(sep)[-1].split("-")[4]
        epochs = read_epochs(epo, verbose=False)
        epo_type = epo.split(sep)[-1].split("-")[5]

        beh_path = [i for i in beh_paths if numero + '-' + epo_type + '-beh' in i][0]
        eve_path = [i for i in eve_paths if numero + '-eve' in i][0]
        print("INPUT EVENT FILE:", eve_path)
        print("INPUT BEHAV FILE:", beh_path)

        # Drop no responses
        beh = pd.read_csv(beh_path)
        rej_idx=np.where(beh.response==0)[0]
        rej_idx = rej_idx[rej_idx < len(epochs)]
        epochs=epochs.drop(rej_idx)

        ch_thr = compute_thresholds(
            epochs,
            random_state=42,
            method="bayesian_optimization",
            verbose="progressbar",
            n_jobs=-1,
            augment=False
        )
        # save the thresholds in JSON
        ch_list = list(ch_thr.keys())
        ch_list.sort()
        results = np.zeros((len(ch_list), len(epochs)))
        results = results - 1
        for ix, ch in enumerate(ch_list):
            thr = ch_thr[ch]
            ch_tr = epochs.copy().pick_channels([ch]).get_data()
            res = [np.where(ch_tr[i][0] > thr)[0].shape[0] for i in range(len(epochs))]
            res = np.array(res)
            results[ix, :] = res
        name = "{}-{}-{}-{}".format(subject_id, session_id, numero, epo_type)
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
        ax.set_xticklabels([str(i) for i in range(1, len(epochs)+1)])
        ax.set_yticks(list(range(len(ch_list))))
        ax.set_yticklabels(ch_list)
        ax.grid(color='w', linestyle='-', linewidth=0.2)
        ax.set_title(name)
        plt.savefig(
            img_path,
            bbox_inches="tight"
        )
        plt.close("all")