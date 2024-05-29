import sys
import json
import mne
import os
import os.path as op
import subprocess as sp
from utilities import files
import matplotlib.pylab as plt

# parsing command line arguments
try:
    index = int(sys.argv[1])
except:
    print("incorrect subject index")
    sys.exit()

try:
    session_index = int(sys.argv[2])
except:
    print("incorrect file index")
    sys.exit()

try:
    run_index = int(sys.argv[3])
except:
    print("incorrect file index")
    sys.exit()

try:
    json_file = sys.argv[4]
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

subjects = files.get_folders(proc_path,'sub-','')[2]
subjects.sort()
subject = subjects[index]
subject_id = subject.split("/")[-1]
print("ID:", subject_id)

sub_path = op.join(proc_path, subject_id)
files.make_folder(sub_path)

sessions = files.get_folders(subject,'ses','')[2]
sessions.sort()
session=sessions[session_index]
session_id = session.split("/")[-1]

meg_path = op.join(subject, session_id, "meg")

qc_folder = op.join(sub_path, session_id, "QC")
files.make_folder(qc_folder)

raw_paths = files.get_files(session, "zapline-" + subject_id+"-"+session_id, "-raw.fif")[2]
raw_paths.sort()
raw_path = raw_paths[run_index]

event_paths = files.get_files(session, subject_id+"-"+session_id, "-eve.fif")[2]
event_paths.sort()
event_path = event_paths[run_index]

ica_paths = files.get_files(session, subject_id+"-"+session_id, "-ica.fif")[2]
ica_paths.sort()
ica_path = ica_paths[run_index]

ica_json_file = op.join(
    session,
    "{}-{}-ICA_to_reject.json".format(subject_id,session_id)
)


print("SUBJ: {}".format(subject_id), session_index, run_index)
print("INPUT RAW FILE:", raw_path.split(os.sep)[-1])
print("INPUT EVENT FILE:", event_path.split(os.sep)[-1])
print("INPUT ICA FILE:", ica_path.split(os.sep)[-1])
print("INPUT JSON FILE", ica_json_file.split(os.sep)[-1])

raw = mne.io.read_raw_fif(
    raw_path, preload=True, verbose=False
)

events = mne.read_events(event_path)

ica = mne.preprocessing.read_ica(
    ica_path, verbose=False
)

raw.crop(
    tmin=raw.times[events[0,0]],
    tmax=raw.times[events[-1,0]]
)
raw.filter(1,20, verbose=False)
raw.close()

sp.Popen(
    ["mousepad", str(ica_json_file)],
    stdout=sp.DEVNULL,
    stderr=sp.DEVNULL
)
print('')

title_ = "sub:{}, session:{}, file: {}".format(subject_id, session_id, ica_path.split(os.sep)[-1])

ica.plot_components(inst=raw, show=False, title=title_)

ica.plot_sources(inst=raw, show=False, title=title_)

plt.show()

