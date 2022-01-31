import sys
import json
from utilities import files
import os.path as op
import mne
import numpy as np
import matplotlib.pyplot as plt

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

for session in [sessions[5]]:
    session_id = session.split("/")[-1]

    meg_path = op.join(session, "meg")

    sess_path = op.join(sub_path, session_id)
    files.make_folder(sess_path)

    qc_folder = op.join(sess_path, "QC")
    files.make_folder(qc_folder)

    raw_paths = files.get_files(sess_path, subject_id + "-" + session_id, "-raw.fif")[2]
    raw_paths.sort()

    for raw_path in raw_paths:

        numero = str(raw_path.split("-")[-2]).zfill(3)

        raw = mne.io.read_raw_fif(raw_path, verbose=False)
        raw_events = mne.find_events(
            raw,
            stim_channel="UPPT001",
            min_duration=0.002,
            verbose="DEBUG",
            consecutive=True
        )
        n_evts_to_remove = 0
        trial_evts = np.where(raw_events[:, 2] == 10)[0]
        fix_evts = np.where(raw_events[:, 2] == 20)[0]
        dots_evts = np.where(raw_events[:, 2] == 30)[0]
        del_evts = np.where(raw_events[:, 2] == 40)[0]
        instr_evts = np.where(raw_events[:, 2] == 50)[0]
        resp_evts = np.where(raw_events[:, 2] == 60)[0]
        iti_evts = np.where(raw_events[:, 2] == 70)[0]
        if len(trial_evts) > 180:
            n_evts_to_remove += 1
        if len(fix_evts) > 180:
            n_evts_to_remove += 1
        if len(dots_evts) > 180:
            n_evts_to_remove += 1
        if len(del_evts) > 180:
            n_evts_to_remove += 1
        if len(instr_evts) > 180:
            n_evts_to_remove += 1
        if len(resp_evts) > 180:
            n_evts_to_remove += 1
        if len(iti_evts) > 180:
            n_evts_to_remove += 1

        raw_events = raw_events[n_evts_to_remove + 1:, :]

        evt_idx = 0
        while evt_idx < raw_events.shape[0]:
            # instruction cue and no resp
            if raw_events[evt_idx, 2] == 50 and raw_events[evt_idx + 1, 2] == 70:
                instr_time = raw_events[evt_idx, 0]
                iti_time = raw_events[evt_idx + 1, 0]
                resp_time = int(instr_time + .5 * (iti_time - instr_time))
                raw_events = np.insert(raw_events, evt_idx + 1, [resp_time, 0, 60], axis=0)
            evt_idx += 1

        trial_evts = np.where(raw_events[:, 2] == 10)[0]
        fix_evts = np.where(raw_events[:, 2] == 20)[0]
        dots_evts = np.where(raw_events[:, 2] == 30)[0]
        del_evts = np.where(raw_events[:, 2] == 40)[0]
        instr_evts = np.where(raw_events[:, 2] == 50)[0]
        resp_evts = np.where(raw_events[:, 2] == 60)[0]
        iti_evts = np.where(raw_events[:, 2] == 70)[0]
        print('{} trial events'.format(len(trial_evts)))
        print('{} fixation events'.format(len(fix_evts)))
        print('{} dots events'.format(len(dots_evts)))
        print('{} delay events'.format(len(del_evts)))
        print('{} instruction cue events'.format(len(instr_evts)))
        print('{} response events'.format(len(resp_evts)))
        print('{} iti events'.format(len(iti_evts)))

        diode_times=[]

        for adc_chan_idx in range(1,5):
            diode_ch_name='UADC00%d' % adc_chan_idx
            if diode_ch_name in raw.ch_names:
                diode,times=raw[diode_ch_name,:]
                n_times=len(times)
                idx=range(raw_events[fix_evts[0],0],raw_events[iti_evts[-1],0]+1)
                diode[0,:]=(diode[0,:]-np.min(diode[0,idx]))/(np.max(diode[0,idx])-np.min(diode[0,idx]))
                #poss_threshs=np.arange(np.median(diode[0,idx]), np.max(diode[0,idx]), 0.05)
                #for diode_thresh in poss_threshs:
                diode_thresh=0.8
                diode_up_down=(diode[0,:]>diode_thresh).astype(int)
                diode_up_down[0:raw_events[fix_evts[0],0]]=0
                diode_up_down[raw_events[iti_evts[-1],0]:]=0
                diode_diff = np.diff(diode_up_down)

                # diode_up_times = np.where(diode_diff > 0)[0]
                # diode_down_times = np.where(diode_diff < 0)[0]
                # if diode_down_times[0]<diode_up_times[0]:
                #     diode_down_times=diode_down_times[1:]
                # if diode_up_times[-1]>diode_down_times[-1]:
                #     diode_up_times=diode_up_times[:-1]
                # diode_durations=(diode_down_times-diode_up_times)/raw.info['sfreq']
                # diode_times=diode_up_times[(diode_durations>1.9) & (diode_durations<2)]
                diode_times=np.where(diode_diff>0)[0]
                if len(diode_times) == 180 or len(diode_times) == 360:
                    fig = plt.figure()
                    plt.plot(times, diode[0, :], 'b')
                    plt.plot([times[0], times[-1]], [diode_thresh, diode_thresh], 'r')
                    for diode_time in diode_times:
                        plt.plot([times[diode_time], times[diode_time]], [np.min(diode, axis=1), diode_thresh], 'g')
                    plt.savefig(
                        op.join(qc_folder, "{}-{}-{}-diode_{}.png".format(subject_id, session_id, numero, diode_ch_name))
                    )
                    plt.close("all")
                    if len(diode_times)==180 or len(diode_times)==360:
                        break
                else:
                    diode_times=[]

        num_diode_onsets = len(diode_times)
        print('num diode onsets=%d' % num_diode_onsets)

        # In early sessions there was only a diode signal for the dots
        dots_onset = diode_times
        # In later sessions there was a diode for the dots and the instruction stimulus
        if len(diode_times) > 180:
            dots_onset = diode_times[0::2]
            instr_onset = diode_times[1::2]

        trial_idx = 0
        for i in range(raw_events.shape[0]):
            # Correct dots onset
            if raw_events[i,2] == 30:
                # Use photodiode signal if exists
                if len(diode_times) == 180 or len(diode_times) == 360:
                    raw_events[i,0] = dots_onset[trial_idx]
                # Use mean delay if not
                else:
                    raw_events[i,0] = raw_events[i,0] + int(0.0301856*raw.info['sfreq'])
            # Correct instruction onset
            elif raw_events[i,2] == 50:
                # Use photodiode signal if exists
                if len(diode_times) == 360:
                    raw_events[i,0] = instr_onset[trial_idx]
                # Use mean delay if not
                else:
                    # Mean delay is different for first trial
                    if trial_idx == 0:
                        raw_events[i, 0] = raw_events[i, 0] + int(0.0190278 * raw.info['sfreq'])
                    else:
                        raw_events[i, 0] = raw_events[i, 0] + int(0.0302645 * raw.info['sfreq'])
                trial_idx = trial_idx + 1

        raw, events = raw.copy().resample(
            sfreq,
            npad="auto",
            events=raw_events,
            n_jobs=4,
        )

        assert (events[-1, 2] == 70)

        raw_path = op.join(
            sess_path,
            "{}-{}-{}-raw.fif".format(subject_id, session_id, numero)
        )
        eve_path = op.join(
            sess_path,
            "{}-{}-{}-eve.fif".format(subject_id, session_id, numero)
        )

        raw.save(
            raw_path,
            fmt="single",
            overwrite=True)

        print("RAW SAVED:", raw_path)

        raw.close()

        mne.write_events(
            eve_path,
            events
        )

        print("EVENTS SAVED:", eve_path)
