import sys
import json
import os.path as op

from utilities import files
import matlab.engine
from os import sep
import matplotlib.pyplot as plt
import numpy as np

def run(index, json_file, parasite):
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

        localizer_paths = files.get_files(op.join(session, 'spm'), "invert_fmspm_converted_autoreject", "-epo_localizer_results.json")[2]
        localizer_paths.sort()

        for localizer_path in localizer_paths:
            epo_type = localizer_path.split(sep)[-1].split("-")[6]
            numero = localizer_path.split(sep)[-1].split("-")[5]

            with open(localizer_path) as results_file:
                localizer_results = json.load(results_file)

            fig = plt.figure(figsize=(18,6))
            peak_idx = localizer_results['peak_idx']

            res_fname = parasite.invert_tc(path, subject_id, session_id,
                                           numero, epo_type, peak_idx+1,
                                           nargout=1)

            with open(res_fname) as results_file:
                results = json.load(results_file)

            t_g0 = np.where(np.array(localizer_results['times']) > 0)[0]
            p_source=np.array(localizer_results['pial_source'])
            max_idx = np.argmax(np.abs(p_source[t_g0]))
            if p_source[t_g0][max_idx] > 0:
                p_source = -1 * p_source
            w_source = np.array(localizer_results['white_source'])
            max_idx = np.argmax(np.abs(w_source[t_g0]))
            if w_source[t_g0][max_idx] > 0:
                w_source = -1 * w_source

            ax = plt.subplot(1, 2, 1)
            plt.plot(localizer_results['times'], p_source)
            plt.plot(localizer_results['times'], w_source)
            plt.legend(['pial', 'white'])
            plt.xlabel('Time (s)')
            plt.ylabel('Source density (pAm/mm^2)')

            ax = plt.subplot(1, 2, 2)
            inner_times=results['times'][results['left_idx']-1:results['right_idx']]
            plt.plot(inner_times, results['f_diff'])
            plt.xlabel('Time (s)')
            plt.ylabel('\Delta F')

            [base, ext] = op.splitext(res_fname)
            fig.suptitle("{}-{}-{}".format(subject_id, session_id, numero))
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

    parasite = matlab.engine.start_matlab()

    run(index, json_file, parasite)