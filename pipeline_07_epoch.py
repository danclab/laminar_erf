import sys
import json
import mne
import os.path as op
import shutil
import pandas as pd
import numpy as np
from mne import read_epochs

from utilities import files

def run(index, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]
    hi_pass = parameters["high_pass_filter"]
    low_pass = parameters["low_pass_filter"]

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

        out_sess_path = op.join(sub_path, session_id)
        files.make_folder(out_sess_path)

        qc_folder = op.join(out_sess_path, "QC")
        files.make_folder(qc_folder)

        raw_paths = files.get_files(sess_path, "zapline-" + subject_id + "-" + session_id, "-raw.fif")[2]
        raw_paths.sort()

        ica_json_file = op.join(
            sess_path,
            "{}-{}-ICA_to_reject.json".format(subject_id, session_id)
        )

        with open(ica_json_file) as ica_file:
            ica_files = json.load(ica_file)

        ica_keys = list(ica_files.keys())
        ica_keys.sort()

        event_paths = files.get_files(sess_path, subject_id + "-" + session_id, "-eve.fif")[2]
        event_paths.sort()

        raw_ica_eve = list(zip(raw_paths, ica_keys, event_paths))

        for (raw_path, ica_key, eve_path) in raw_ica_eve:
            # for (raw_path, ica_key, eve_path) in [raw_ica_eve[3]]:
            ica_path = op.join(
                sess_path,
                ica_key
            )
            numero = str(raw_path.split("-")[-2]).zfill(3)

            behav_path = op.join(
                sess_path, "{}-{}-{}-beh.csv".format(subject_id, session_id, numero)
            )

            print("INPUT RAW FILE:", raw_path)
            print("INPUT EVENT FILE:", eve_path)
            print("INPUT ICA FILE:", ica_path)
            print("INPUT BEHAV FILE:", behav_path)

            ica_exc = ica_files[ica_key]

            events = mne.read_events(eve_path)

            ica = mne.preprocessing.read_ica(
                ica_path,
                verbose=False
            )

            raw = mne.io.read_raw_fif(
                raw_path,
                verbose=False,
                preload=True
            )

            raw = ica.apply(
                raw,
                exclude=ica_exc,
                verbose=False
            )

            raw.filter(
                l_freq=hi_pass,
                h_freq=low_pass
            )

            epochs_dict = {
                "visual": [30, -1, 3.5],
                "motor": [60, -2, 2]
            }

            for i in epochs_dict.keys():
                trig, tmin, tmax = epochs_dict[i]
                epoch = mne.Epochs(
                    raw,
                    mne.pick_events(events, include=trig),
                    tmin=tmin,
                    tmax=tmax,
                    baseline=None,
                    verbose=True,
                    detrend=1
                )

                epoch_path = op.join(
                    out_sess_path,
                    "{}-{}-{}-{}-epo.fif".format(subject_id, session_id, numero, i)
                )

                epoch.save(
                    epoch_path,
                    fmt="double",
                    overwrite=True,
                    verbose=False,
                )

                beh = pd.read_csv(behav_path)
                n_trials = len(epoch)
                if len(beh) > n_trials:
                    beh = beh.drop(axis=0, index=list(range(n_trials, len(beh))))
                epoch_behav_path = op.join(
                    out_sess_path,
                    "{}-{}-{}-{}-beh.csv".format(subject_id, session_id, numero, i)
                )
                beh.to_csv(epoch_behav_path)

                print("SAVED:", epoch_path)
                print("SAVED:", epoch_behav_path)


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