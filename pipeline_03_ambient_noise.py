import json
import sys

from meegkit.dss import dss_line_iter

from utilities import files
import os.path as op
import mne
import numpy as np
import matplotlib.pylab as plt

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
sub_path = op.join(path, "raw")
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

    meg_path = op.join(session, "meg")

    sess_path = op.join(sub_path, session_id)
    files.make_folder(sess_path)

    qc_folder = op.join(sess_path, "QC")
    files.make_folder(qc_folder)

    raw_paths = files.get_files(sess_path, subject_id+"-"+session_id, "-raw.fif")[2]
    raw_paths.sort()
    event_paths = files.get_files(sess_path, subject_id+"-"+session_id, "-eve.fif")[2]
    event_paths.sort()

    raw_eve_list = list(zip(raw_paths, event_paths))

    # for raw_path, eve_path in [raw_eve_list[0]]:
    for raw_path, eve_path in raw_eve_list:
        print("INPUT RAW FILE:", raw_path)
        print("EVE_RAW MATCH:", raw_path.split("-")[-2] == eve_path.split("-")[-2])
        numero = str(raw_path.split("-")[-2]).zfill(3)

        raw = mne.io.read_raw_fif(raw_path, verbose=False)
        # raw = raw.apply_gradient_compensation(2, verbose=True)
        raw = raw.pick_types(meg=True, eeg=False, ref_meg=True)
        fig = raw.plot_psd(
            tmax=np.inf, fmax=260, average=True, show=False, picks="meg"
        )
        fig.suptitle("{}-{}-{}".format(subject_id, session_id, numero))
        plt.savefig(
            op.join(qc_folder, "{}-{}-{}-raw-psd.png".format(subject_id, session_id, numero)),
            dpi=150, bbox_inches="tight"
        )
        plt.close("all")

        info = raw.info
        raw = raw.get_data()

        zapped, iterations = dss_line_iter(
            raw.transpose(),
            50.0,
            info['sfreq'],
            win_sz=20,
            spot_sz=5.5,
            show=True,
            prefix="{}/{}-{}-{}-50_iter".format(qc_folder, subject_id, session_id, numero)
        )

        zapped, iterations = dss_line_iter(
            zapped,
            60.0,
            info['sfreq'],
            win_sz=20,
            spot_sz=5.5,
            show=True,
            prefix="{}/{}-{}-{}-60_iter".format(qc_folder, subject_id, session_id, numero)
        )

        raw = mne.io.RawArray(
            zapped.transpose(),
            info
        )

        fig = raw.plot_psd(tmax=np.inf, fmax=260, average=True, show=False)
        fig.suptitle("{}-{}-{}".format(subject_id, session_id, numero))
        plt.savefig(
            op.join(qc_folder, "{}-{}-{}-zapline-raw-psd.png".format(subject_id, session_id, numero)),
            dpi=150,
            bbox_inches="tight"
        )
        plt.close("all")

        f_n = str(numero).zfill(3)  # file number
        out_path = op.join(
            sess_path,
            "zapline-{}-{}-{}-raw.fif".format(subject_id, session_id, f_n)
        )

        raw.save(
            out_path,
            overwrite=True
        )
        del raw