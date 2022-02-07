import sys
import json
import os.path as op
from utilities import files
import matlab.engine
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

    subjects = files.get_folders_files(proc_path)[0]
    subjects.sort()
    subject = subjects[index]
    subject_id = subject.split("/")[-1]
    print("ID:", subject_id)

    sessions = files.get_folders(subject, 'ses', '')[2]
    sessions.sort()

    parasite = matlab.engine.connect_matlab()

    for session in sessions:
        spm_paths = files.get_files(session, "spm_converted", ".mat")[2]
        spm_paths.sort()

        for spm_path in spm_paths:
            print(spm_path)
            parasite.epoch_average(spm_path, nargout=0)


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