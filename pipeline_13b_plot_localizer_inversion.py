import sys
import json
import os.path as op

from mne import read_epochs

from extra.tools import dump_the_dict
from pipeline_13_localizer_inversion import plot_localizer_results
from utilities import files
from os import sep
import nibabel as nb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_surface_plotting import plot_surf

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

    for session in sessions:
        session_id = session.split("/")[-1]

        epo_paths = files.get_files(session, subject_id + "-" + session_id + "-001", "-epo.fif")[2]
        epo_types = []
        for epo in epo_paths:
            epo_types.append(epo.split(sep)[-1].split("-")[5])

        for epo_type in epo_types:
            res_fnames = files.get_files(op.join(session, 'spm'), "invert_fmspm_converted_autoreject", epo_type + "-epo_localizer_results.json")[2]
            res_fnames.sort()

            for run_idx,res_fname in enumerate(res_fnames):
                numero = res_fname.split(sep)[-1].split("-")[5]

                with open(res_fname) as results_file:
                    results = json.load(results_file)

                fig = plot_localizer_results(subject_id, session_id, numero, epo_type, results)
                [base, ext] = op.splitext(res_fname)
                plt.savefig(
                    '{}.png'.format(base),
                    dpi=300,
                    pad_inches=0,
                    transparent=False
                )
                plt.close("all")





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