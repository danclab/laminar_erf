import sys
import json
import os.path as op

from mne import read_epochs

from utilities import files
from lameg.util import ctf_fif_spm_conversion
import spm_standalone
from os import sep

epo_types = ['motor', 'visual']

def run(index, json_file, spm):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]

    der_path = op.join(path, "derivatives")
    files.make_folder(der_path)
    proc_path = op.join(der_path, "processed")
    files.make_folder(proc_path)

    subjects = files.get_folders(proc_path, 'sub-', '')[2]
    subjects.sort()
    subject = subjects[index]
    subject_id = subject.split("/")[-1]
    print("ID:", subject_id)

    raw_meg_dir = op.join(parameters["dataset_path"], "raw")

    sessions = files.get_folders(subject, 'ses', '')[2]
    sessions.sort()

    for session in sessions:
        session_id = session.split("/")[-1]

        spm_path = op.join(session, "spm")
        files.make_folder(spm_path)

        raw_meg_path = op.join(raw_meg_dir, subject_id, session_id, "meg")
        ds_paths = files.get_folders_files(raw_meg_path)[0]
        ds_paths = [i for i in ds_paths if "misc" not in i]
        ds_paths.sort()
        res4_paths = [files.get_files(i, "", ".res4")[2][0] for i in ds_paths]
        res4_paths.sort()

        for epo_type in epo_types:
            fif_paths = files.get_files(session, "sub", epo_type + "-epo.fif")[2]

            fif_paths.sort()

            for fif in fif_paths:
                path_split = fif.split(sep)
                filename_core = path_split[-1].split(".")[0]
                ctf_fif_spm_conversion(
                    fif,
                    res4_paths[0],
                    spm_path,
                    True,
                    prefix='spm_converted_',
                    spm_instance=spm
                )

                average_file = op.join(session, filename_core + "-ave.fif")
                epochs = read_epochs(fif, verbose=False)
                epochs = epochs.average()
                epochs.save(average_file, overwrite=True)
                ctf_fif_spm_conversion(
                    average_file,
                    res4_paths[0],
                    spm_path,
                    False,
                    prefix='spm_converted_',
                    spm_instance=spm
                )


if __name__ == '__main__':
    spm = spm_standalone.initialize()
    json_file = "settings.json"
    for index in range(8):
        run(index, json_file, spm)
    spm.terminate()