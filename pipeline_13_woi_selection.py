import sys
import json
import os.path as op

import mne
import numpy as np
from mne.viz.utils import _find_peaks
import matplotlib.pyplot as plt
from scipy.signal import argrelmax

from extra.tools import dump_the_dict
from utilities import files
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

    epo_paths = files.get_files(sessions[0], subject_id + "-" + sessions[0].split("/")[-1] + "-001", "-epo.fif")[2]
    epo_types = []
    for epo in epo_paths:
        epo_types.append(epo.split(sep)[-1].split("-")[5])
    epo_types.sort()

    qc_folder = op.join(subject, "QC")
    files.make_folder(qc_folder)


    for epo_type in epo_types:
        name = "{}-{}".format(subject_id, epo_type)

        all_data = []

        for session in sessions:
            fif_paths = files.get_files(session, "autoreject-sub", epo_type + "-epo.fif")[2]
            fif_paths.sort()

            for fif_path in fif_paths:
                epoch = mne.read_epochs(fif_path)
                evoked = epoch.average().filter(None, 30, fir_design='firwin')
                all_data.append(evoked.data)
        min_size=1000000
        for data in all_data:
            if data.shape[0]<min_size:
                min_size=data.shape[0]
        for idx in range(len(all_data)):
            all_data[idx]=all_data[idx][:min_size,:]
        all_data=np.array(all_data)
        all_data=np.squeeze(np.mean(all_data, axis=0))
        evoked.data=all_data

        #ind = np.argpartition(np.std(evoked.data, axis=1), -5)[-5:]
        #chans=[epoch.ch_names[x] for x in ind]
        woi_picks = mne.pick_types(epoch.info, meg=True, ref_meg=False)
        if 'motor' in epo_type:
            woi_picks = mne.pick_channels(epoch.ch_names, ['MLT22', 'MLT12', 'MLT33', 'MLT23', 'MLT13', 'MLP34',
                                                           'MLP44', 'MLP56', 'MLP55', 'MLP43', 'MLP33', 'MLP42',
                                                           ])
        elif 'visual' in epo_type:
            woi_picks = mne.pick_channels(epoch.ch_names, ['MLO32','MLO22','MLO23','MLO33','MLO43','MLO42','MLO31',
                                                           'MRO32','MRO22','MRO23','MRO33','MRO43','MRO42','MRO31'])
        #woi_picks = mne.pick_channels(epoch.ch_names, chans)
        woi_evoked = all_data[woi_picks,:]

        npeaks=5
        #gfp = np.mean(np.abs(woi_evoked), axis=0)
        gfp = np.std(woi_evoked.data, axis=0)
        #gfp = np.std(woi_evoked.data[:,t_idx], axis=0)
        #order = len(evoked.times) // 30
        #if order < 1:
        order = 1
        peaks = argrelmax(gfp, order=order, axis=0)[0]
        if len(peaks) > npeaks:
            max_indices = np.argsort(gfp[peaks])[-npeaks:]
            peaks = np.sort(peaks[max_indices])
        peaks=peaks[gfp[peaks]>np.mean(gfp)+np.std(gfp)]
        peak_times = evoked.times[peaks]
        if len(peak_times) == 0:
            peak_times = [evoked.times[gfp.argmax()]]

        #peaks=_find_peaks(evoked, 3)
        if 'motor' in epo_type:
            within_peaks=np.where((peak_times>-.1) & (peak_times<.1))[0]
            within_peak_times=peak_times[within_peaks]
            peak = within_peak_times[np.argmax(gfp[peaks[within_peaks]])]
        else:
            peak=np.min(peak_times[peak_times>0])
        woi=[peak-.02, peak+0.02]

        fig=plt.figure()
        plt.plot(evoked.times, gfp)
        plt.plot([peak, peak],plt.ylim(),'r--')
        plt.title(name)
        plt.show()

        fig = evoked.plot_joint(show=False)
        yl=fig.axes[0].get_ylim()
        fig.axes[0].plot([woi[0], woi[0]],yl,'r--')
        fig.axes[0].plot([woi[1], woi[1]],yl, 'r--')
        fig.axes[0].set_ylim(yl)
        plt.savefig(op.join(qc_folder, "{}-epo-WOI.png".format(name)))
        plt.close("all")

        woi_path = op.join(subject, '{}-epo-woi.json'.format(name))
        dump_the_dict(woi_path, {'woi': woi})


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