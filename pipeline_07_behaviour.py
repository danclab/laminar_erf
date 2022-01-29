import sys
import json
import numpy as np
import pandas as pd
import os
import os.path as op

import scipy.io

from utilities import visang, files

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
sub_path = op.join(path, "raw")

der_path = op.join(path, "derivatives")
files.make_folder(der_path)
proc_path = op.join(der_path, "processed")
files.make_folder(proc_path)

subjects = files.get_folders_files(sub_path)[0]
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

    beh_path = op.join(op.join(subject, session_id), "behaviour")

    beh_files = files.get_files(beh_path, subject_id+"_"+session_id+"_block", "data.mat")[2]
    beh_files.sort()

    stim_files = files.get_files(beh_path, subject_id + "_" + session_id + "_block", "stim.mat")[2]
    stim_files.sort()

    qc_folder = op.join(sess_path, "QC")
    files.make_folder(qc_folder)

    # for beh_file in [beh_files[4]]:
    for beh_file, stim_file in zip(beh_files, stim_files):
        numero=beh_file.split(os.sep)[-1].split("_")[-2].split("-")[1].zfill(3)

        raw_behav = scipy.io.loadmat(beh_file)
        print(stim_file)
        raw_stim = scipy.io.loadmat(stim_file)

        responses=raw_behav['data'][0,0][20]
        trials=raw_stim['stim'][0,0][1]

        trial_n = trials.shape[0]

        # Correct stim direction mismatch
        trials[:,0]=1+2-trials[:,0]
        trials[:,3]=1+2-trials[:,3]

        coherence_levels=np.sort(np.unique(trials[:,1]))

        congruence = []
        coherence = []
        response = [] # L=1, R=2
        rt = []
        correct = []

        for tr_ix in range(trial_n):

            if trials[tr_ix,2]==0:
                congruence.append('incongruent')
            else:
                congruence.append('congruent')

            if trials[tr_ix,1]==coherence_levels[0]:
                coherence.append('low')
            elif trials[tr_ix,1]==coherence_levels[1]:
                coherence.append('med')
            else:
                coherence.append('high')

            response.append(responses[tr_ix,0])
            rt.append(responses[tr_ix,1])
            correct.append(trials[tr_ix,3]==responses[tr_ix,0])

        data = {
            "subject_id": subject_id,
            "session": session_id,
            "block": numero,
            "trial": np.arange(trial_n),
            "trial_congruence": np.array(congruence),
            "trial_coherence": np.array(coherence),
            "response": np.array(response),
            "rt": np.array(rt),
            "correct": np.array(correct).astype(int),
        }

        data = pd.DataFrame.from_dict(data)


        data_path = op.join(
            sess_path, "{}-{}-{}-beh.csv".format(subject_id, session_id, numero)
        )

        data.to_csv(data_path, index=False)