import matlab.engine
import os.path as op
import json
from utilities import files
from os import sep

index=0
json_file = "settings.json"
parasite = matlab.engine.start_matlab()

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
session=sessions[0]

spm_path = op.join(session,'spm','fmspm_converted_autoreject-sub-001-ses-01-001-motor-epo.mat')
epo_path = op.join(session, 'autoreject-sub-001-ses-01-001-motor-epo.fif')

epo_type = spm_path.split(sep)[-1].split("-")[6]
numero = spm_path.split(sep)[-1].split("-")[5]


