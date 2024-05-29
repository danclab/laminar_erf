import sys
import json
import os.path as op
from utilities import files
from lameg.util import fif_spm_conversion
from os import sep

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

        epo_paths = files.get_files(session, subject_id + "-" + session_id + "-001", "-epo.fif")[2]
        epo_types = []
        for epo in epo_paths:
            epo_types.append(epo.split(sep)[-1].split("-")[5])

        for epo_type in epo_types:
            fif_paths = files.get_files(session, "autoreject-sub", epo_type + "-epo.fif")[2]

            fif_paths.sort()

            fif_res4_paths = list(zip(fif_paths, res4_paths))
            for fif, res4 in fif_res4_paths:
                print(fif, res4)

                fif_spm_conversion(
                    fif, res4, spm_path,
                    prefix="spm_converted_",
                    epoched=True
                )



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