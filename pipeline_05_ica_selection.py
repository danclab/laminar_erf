import sys
import json
import numpy as np
import pandas as pd
import subprocess
import mne
import os.path as op
from utilities import files
from pyedfread import edf
from extra.tools import update_key_value, dump_the_dict, resamp_interp
from ecgdetectors import Detectors
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import gc

# Time lagged cross correlation
def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

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

mne.set_log_level(verbose=None)

path = parameters["dataset_path"]
sfreq = parameters["downsample_dataset"]

der_path = op.join(path, "derivatives")
files.make_folder(der_path)
proc_path = op.join(der_path, "processed")
files.make_folder(proc_path)

subjects = files.get_folders_files(proc_path)[0]
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

    beh_path = op.join(path, "raw", subject_id, session_id, "behaviour")

    edf_paths = files.get_files(beh_path, "", ".edf")[2]
    edf_paths.sort()
    edf_paths = [i for i in edf_paths if "block" in i]

    sess_path = op.join(sub_path, session_id)
    files.make_folder(sess_path)

    qc_folder = op.join(sess_path, "QC")
    files.make_folder(qc_folder)

    raw_paths = files.get_files(sess_path, "zapline-" + subject_id +"-"+session_id, "-raw.fif")[2]
    raw_paths.sort()

    event_paths = files.get_files(sess_path, subject_id + "-" + session_id, "-eve.fif")[2]
    event_paths.sort()

    ica_json_file = op.join(
        sess_path,
        "{}-{}-ICA_to_reject.json".format(subject_id, session_id)
    )

    with open(ica_json_file) as ica_file:
        ica_files = json.load(ica_file)

    ica_keys = list(ica_files.keys())
    ica_keys.sort()

    raw_ica_edf = list(zip(raw_paths, event_paths, ica_keys, edf_paths))

    ecg_out = dict()
    eog_out = dict()

    eog_file_path = op.join(
        sess_path,
        "{}-{}-eog-stats.json".format(subject_id, session_id)
    )
    ecg_file_path = op.join(
        sess_path,
        "{}-{}-ecg-stats.json".format(subject_id, session_id)
    )

    ds = Detectors(sfreq)

    for (raw_path, event_path, ica_key, edf_path) in raw_ica_edf:
        ica_path = op.join(
            sess_path,
            ica_key
        )
        numero = str(raw_path.split("-")[-2]).zfill(3)

        print("INPUT RAW FILE:", raw_path)
        print("INPUT EVENT FILE:", event_path)
        print("INPUT ICA FILE:", ica_path)
        print("INPUT EDF FILE:", edf_path)

        raw = mne.io.read_raw_fif(
            raw_path,
            verbose=False,
            preload=True
        )

        events = mne.read_events(event_path)

        ica = mne.preprocessing.read_ica(
            ica_path,
            verbose=False
        )
        raw.crop(
            tmin=raw.times[events[0, 0]],
            tmax=raw.times[events[-1, 0]]
        )
        raw.filter(1, 20)

        raw.close()

        ica_com = ica.get_sources(raw)
        raw = None
        gc.collect()
        ica_times = ica_com.times
        ica_data = ica_com.get_data()
        ica_com.close()
        ica_com = None
        gc.collect()

        # https://github.com/berndporr/py-ecg-detectors
        # variance of the distance between detected R peaks
        # if the variance is not distinct enough from the 1 percentile,
        # signal has to be found manually, indicated as 666 in the first item
        # of the list.
        r_hr = [ds.hamilton_detector(ica_data[i]) for i in range(ica_data.shape[0])]
        r_hr_var = [np.var(np.diff(i)) for i in r_hr]
        ecg_out[ica_key] = r_hr_var
        r_hr_var = np.array(r_hr_var)

        if (np.percentile(r_hr_var, 1) - np.min(r_hr_var)) > 500:
            hr = list(np.where(r_hr_var < np.percentile(r_hr_var, 1))[0])

            for h in hr:
                fig=plt.figure()
                fig.suptitle("{}-{}-{}".format(subject_id, session_id, numero))
                plt.plot(ica_data[h])
                plt.plot(r_hr[h], ica_data[h][r_hr[h]], 'ro')
                plt.title("Detected R peaks")
                plt.savefig(
                    op.join(qc_folder, "{}-{}-{}-hr_comp_{}.png".format(subject_id, session_id, numero, h)),
                    dpi=150,
                    bbox_inches="tight"
                )
                plt.close("all")
        else:
            hr = [666]

        samples, events, messages = edf.pread(edf_path)
        eye = ["left", "right"][events.eye.unique()[0]]

        del events
        del messages

        # cleaning the eyetracking datas
        samples = samples.loc[samples.time != 0.0]
        # start = samples.loc[samples.input == 252.]
        # end = samples.loc[samples.input == 253.]
        # start_ix = start.index[0] - 100
        # end_ix = end.index[-1] + 100
        # samples = samples.iloc[start_ix:end_ix]
        samples.reset_index(inplace=True)
        samples.time = samples.time - samples.time.iloc[0]

        # picking the relevant piece of data
        gx = samples["gx_{}".format(eye)]
        gy = samples["gy_{}".format(eye)]
        samples_times = samples.time / 1000

        del samples

        # resampling to meg sampling rate
        gx = resamp_interp(samples_times, gx, ica_times)
        gy = resamp_interp(samples_times, gy, ica_times)

        # gx, gy is a gaze screen position, EyeLink recorded blinks as position way
        # outside of the screen, thus safe threshold to detect blinks.
        # dependent on the screen resolution.
        blink_ix = np.where(gy > 600)[0]

        clean_gx = np.copy(gx)
        clean_gy = np.copy(gy)
        gx_iqr = np.percentile(gx, [25, 50])
        gy_iqr = np.percentile(gx, [25, 50])
        gx_iqr_med = np.median(gx[np.where((gx > gx_iqr[0]) & (gx < gx_iqr[1]))[0]])
        gy_iqr_med = np.median(gy[np.where((gy > gy_iqr[0]) & (gy < gy_iqr[1]))[0]])

        clean_gx[blink_ix] = gx_iqr_med
        clean_gy[blink_ix] = gy_iqr_med

        clean_gx = pd.Series(clean_gx).interpolate()
        clean_gy = pd.Series(clean_gy).interpolate()
        gx = pd.Series(gx)
        gy = pd.Series(gy)

        # x, y, clean_x, clean_y
        # ICA comp N x 4x(r, p)
        ica_eog = []
        comp = []
        for i in range(ica_data.shape[0]):
            out=[]
            for j in [gx, gy, clean_gx, clean_gy]:
                lags = np.arange(-(20000), (20000), 10)  # contrained
                rs = np.nan_to_num([crosscorr(pd.Series(ica_data[i]), j, lag) for lag in lags])
                #out.append(pearsonr(j, ica_data[i]))
                out.append(np.max(rs))
            comp.append(out)
            results = np.array(out)
            if np.mean(np.abs(results) > 0.15) >= 0.25:
                ica_eog.append(i)

                fig = plt.figure()
                fig.suptitle("{}-{}-{}".format(subject_id, session_id, numero))
                fig.add_subplot(2,2,1)
                plt.plot(gx,ica_data[i],'.')
                plt.xlabel('gx')
                plt.ylabel('comp {}'.format(i))
                fig.add_subplot(2, 2, 2)
                plt.plot(gy,ica_data[i],'.')
                plt.xlabel('gy')
                plt.ylabel('comp {}'.format(i))
                fig.add_subplot(2,2,3)
                plt.plot(clean_gx,ica_data[i],'.')
                plt.xlabel('clean_gx')
                plt.ylabel('comp {}'.format(i))
                fig.add_subplot(2,2,4)
                plt.plot(clean_gy,ica_data[i],'.')
                plt.xlabel('clean_gy')
                plt.ylabel('comp {}'.format(i))
                plt.savefig(
                    op.join(qc_folder, "{}-{}-{}-eog_comp_{}.png".format(subject_id, session_id, numero, i)),
                    dpi=150,
                    bbox_inches="tight"
                )
                plt.close("all")
        eog_out[ica_key] = comp

        # all the numbers have to be integers
        ica_eog.extend(hr)
        ica_eog = [int(i) for i in ica_eog]

        # update of the key values
        update_key_value(ica_json_file, ica_key, ica_eog)

    # dump the stats in json files

    for i in (ecg_file_path, eog_file_path):
        if not op.exists(i):
            subprocess.run(["touch", i])

    dump_the_dict(ecg_file_path, ecg_out)
    dump_the_dict(eog_file_path, eog_out)