import sys
import json
from utilities import files
import os.path as op
import mne
import numpy as np
import matplotlib.pyplot as plt

def run(index, json_file):
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
    subjects = files.get_folders(sub_path, 'sub-', '')[2]
    subjects.sort()
    subject = subjects[index]
    subject_id = subject.split("/")[-1]
    print("ID:", subject_id)
    sub_path = op.join(proc_path, subject_id)
    files.make_folder(sub_path)
    sessions = files.get_folders(subject, 'ses', '')[2]
    sessions.sort()
    for session in sessions:
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

            # Get each evt type
            trial_evts = np.where(raw_events[:, 2] == 10)[0]
            fix_evts = np.where(raw_events[:, 2] == 20)[0]
            dots_evts = np.where(raw_events[:, 2] == 30)[0]
            del_evts = np.where(raw_events[:, 2] == 40)[0]
            instr_evts = np.where(raw_events[:, 2] == 50)[0]
            resp_evts = np.where(raw_events[:, 2] == 60)[0]
            iti_evts = np.where(raw_events[:, 2] == 70)[0]
            print('Initial')
            print('{} trial events'.format(len(trial_evts)))
            print('{} fixation events'.format(len(fix_evts)))
            print('{} dots events'.format(len(dots_evts)))
            print('{} delay events'.format(len(del_evts)))
            print('{} instruction cue events'.format(len(instr_evts)))
            print('{} response events'.format(len(resp_evts)))
            print('{} iti events'.format(len(iti_evts)))

            # Remove first set of events
            all_stds = []
            prev_idx = -1
            for iti_evt in iti_evts:
                t_diff = np.diff(raw_events[prev_idx + 1:iti_evt + 1, 0])
                all_stds.append(np.std(t_diff))
                prev_idx = iti_evt
            np.percentile(all_stds, 1)
            n_evts_to_remove = 0
            if all_stds[0] < np.percentile(all_stds, .005):
                n_evts_to_remove = iti_evts[0] + 1

            # Remove initial events
            raw_events = raw_events[n_evts_to_remove:, :]

            trial_evts = np.where(raw_events[:, 2] == 10)[0]
            fix_evts = np.where(raw_events[:, 2] == 20)[0]
            dots_evts = np.where(raw_events[:, 2] == 30)[0]
            del_evts = np.where(raw_events[:, 2] == 40)[0]
            instr_evts = np.where(raw_events[:, 2] == 50)[0]
            resp_evts = np.where(raw_events[:, 2] == 60)[0]
            iti_evts = np.where(raw_events[:, 2] == 70)[0]
            print('Removed initial events')
            print('{} trial events'.format(len(trial_evts)))
            print('{} fixation events'.format(len(fix_evts)))
            print('{} dots events'.format(len(dots_evts)))
            print('{} delay events'.format(len(del_evts)))
            print('{} instruction cue events'.format(len(instr_evts)))
            print('{} response events'.format(len(resp_evts)))
            print('{} iti events'.format(len(iti_evts)))

            # Add missing fixation events
            evt_idx = 1
            while evt_idx < raw_events.shape[0]:
                # delay and no dots before
                if (raw_events[evt_idx, 2] == 40) and not (raw_events[evt_idx - 1, 2] == 30):
                    del_time = raw_events[evt_idx, 0] * 1 / raw.info['sfreq']
                    dots_time = int((del_time - 2) * raw.info['sfreq'])
                    raw_events = np.insert(raw_events, evt_idx, [dots_time, 0, 30], axis=0)
                evt_idx += 1

            # Add missing fixation events
            evt_idx = 1
            while evt_idx < raw_events.shape[0]:
                # dots and no fixation before
                if (raw_events[evt_idx, 2] == 30) and not (raw_events[evt_idx - 1, 2] == 20):
                    dots_time = raw_events[evt_idx, 0]*1/raw.info['sfreq']
                    fix_time = int((dots_time-1)*raw.info['sfreq'])
                    raw_events = np.insert(raw_events, evt_idx, [fix_time, 0, 20], axis=0)
                evt_idx += 1

            # Add missing trial events
            evt_idx = 1
            while evt_idx < raw_events.shape[0]:
                # fixation and no trial before
                if (raw_events[evt_idx, 2] == 20) and not (raw_events[evt_idx - 1, 2] == 10):
                    instr_time = raw_events[evt_idx, 0] * 1 / raw.info['sfreq']
                    trial_time = int((instr_time - .005) * raw.info['sfreq'])
                    raw_events = np.insert(raw_events, evt_idx, [trial_time, 0, 10], axis=0)
                evt_idx += 1

            # Add missing response events
            evt_idx = 0
            while evt_idx < raw_events.shape[0]:
                # instruction cue and no resp
                if raw_events[evt_idx, 2] == 50 and not (raw_events[evt_idx + 1, 2] == 60):
                    instr_time = raw_events[evt_idx, 0]
                    raw_events = np.insert(raw_events, evt_idx + 1, [instr_time+1, 0, 60], axis=0)
                evt_idx += 1

            # Add missing ITI events
            evt_idx = 0
            while evt_idx < raw_events.shape[0]:
                # response and no iti
                if raw_events[evt_idx, 2] == 60 and not raw_events[evt_idx + 1, 2] == 70:
                    resp_time = raw_events[evt_idx, 0]
                    raw_events = np.insert(raw_events, evt_idx+1, [resp_time + 1, 0, 70], axis=0)
                evt_idx += 1



            trial_evts = np.where(raw_events[:, 2] == 10)[0]
            fix_evts = np.where(raw_events[:, 2] == 20)[0]
            dots_evts = np.where(raw_events[:, 2] == 30)[0]
            del_evts = np.where(raw_events[:, 2] == 40)[0]
            instr_evts = np.where(raw_events[:, 2] == 50)[0]
            resp_evts = np.where(raw_events[:, 2] == 60)[0]
            iti_evts = np.where(raw_events[:, 2] == 70)[0]
            print('Added missing events')
            print('{} trial events'.format(len(trial_evts)))
            print('{} fixation events'.format(len(fix_evts)))
            print('{} dots events'.format(len(dots_evts)))
            print('{} delay events'.format(len(del_evts)))
            print('{} instruction cue events'.format(len(instr_evts)))
            print('{} response events'.format(len(resp_evts)))
            print('{} iti events'.format(len(iti_evts)))

            # Remove events after last ITI event
            iti_evts = np.where(raw_events[:, 2] == 70)[0]
            raw_events = raw_events[0:iti_evts[-1] + 1, :]

            # Get events of each type
            trial_evts = np.where(raw_events[:, 2] == 10)[0]
            fix_evts = np.where(raw_events[:, 2] == 20)[0]
            dots_evts = np.where(raw_events[:, 2] == 30)[0]
            del_evts = np.where(raw_events[:, 2] == 40)[0]
            instr_evts = np.where(raw_events[:, 2] == 50)[0]
            resp_evts = np.where(raw_events[:, 2] == 60)[0]
            iti_evts = np.where(raw_events[:, 2] == 70)[0]
            print('Final')
            print('{} trial events'.format(len(trial_evts)))
            print('{} fixation events'.format(len(fix_evts)))
            print('{} dots events'.format(len(dots_evts)))
            print('{} delay events'.format(len(del_evts)))
            print('{} instruction cue events'.format(len(instr_evts)))
            print('{} response events'.format(len(resp_evts)))
            print('{} iti events'.format(len(iti_evts)))

            fig = plt.figure()
            plt.plot(raw_events[:, 0], raw_events[:, 2])
            plt.savefig(
                op.join(qc_folder,
                        "{}-{}-{}-events.png".format(subject_id, session_id, numero))
            )
            plt.close("all")

            dt = (raw_events[fix_evts, 0] - raw_events[trial_evts, 0]) * 1 / raw.info['sfreq']
            fig = plt.figure()
            plt.hist(dt, bins=100)
            plt.savefig(
                op.join(qc_folder,
                        "{}-{}-{}-dt_trial-fixation.png".format(subject_id, session_id, numero))
            )
            plt.close("all")

            dt = (raw_events[dots_evts, 0] - raw_events[fix_evts, 0]) * 1 / raw.info['sfreq']
            fig = plt.figure()
            plt.hist(dt, bins=100)
            plt.savefig(
                op.join(qc_folder,
                        "{}-{}-{}-dt_fixation-dots.png".format(subject_id, session_id, numero))
            )
            plt.close("all")

            dt = (raw_events[del_evts, 0] - raw_events[dots_evts, 0]) * 1 / raw.info['sfreq']
            fig = plt.figure()
            plt.hist(dt, bins=100)
            plt.savefig(
                op.join(qc_folder,
                        "{}-{}-{}-dt_dots-delay.png".format(subject_id, session_id, numero))
            )
            plt.close("all")

            dt = (raw_events[instr_evts, 0] - raw_events[del_evts, 0]) * 1 / raw.info['sfreq']
            fig = plt.figure()
            plt.hist(dt, bins=100)
            plt.savefig(
                op.join(qc_folder,
                        "{}-{}-{}-dt_delay-instr.png".format(subject_id, session_id, numero))
            )
            plt.close("all")

            dt = (raw_events[resp_evts, 0] - raw_events[instr_evts, 0]) * 1 / raw.info['sfreq']
            fig = plt.figure()
            plt.hist(dt, bins=100)
            plt.savefig(
                op.join(qc_folder,
                        "{}-{}-{}-dt_instr-resp.png".format(subject_id, session_id, numero))
            )
            plt.close("all")

            dt = (raw_events[iti_evts, 0] - raw_events[resp_evts, 0]) * 1 / raw.info['sfreq']
            fig = plt.figure()
            plt.hist(dt, bins=100)
            plt.savefig(
                op.join(qc_folder,
                        "{}-{}-{}-dt_resp-iti.png".format(subject_id, session_id, numero))
            )
            plt.close("all")

            n_trials = len(iti_evts)

            diode_times = []

            # Find diode channel - can be UADC001-UADC004
            for adc_chan_idx in range(1, 5):
                diode_ch_name = 'UADC00%d' % adc_chan_idx
                if diode_ch_name in raw.ch_names:

                    # Get channel signal
                    diode, times = raw[diode_ch_name, :]
                    n_times = len(times)

                    # Only look from first fixation event to last iti event
                    idx = range(raw_events[fix_evts[0], 0], raw_events[iti_evts[-1], 0] + 1)

                    # Normalize between zero and one based on this period
                    diode[0, :] = (diode[0, :] - np.min(diode[0, idx])) / (
                                np.max(diode[0, idx]) - np.min(diode[0, idx]))

                    # Find right threshold
                    poss_threshs = np.arange(np.median(diode[0, idx]), np.max(diode[0, idx]), 0.05)
                    for diode_thresh in poss_threshs:
                        # Find when diode up or down
                        diode_up_down = (diode[0, :] > diode_thresh).astype(int)
                        # Set to down ouside of trials
                        diode_up_down[0:raw_events[fix_evts[0], 0]] = 0
                        diode_up_down[raw_events[iti_evts[-1], 0]:] = 0
                        # Find changes in up/down state
                        diode_diff = np.diff(diode_up_down)
                        # Times where diode is turning on
                        diode_times = np.where(diode_diff > 0)[0]
                        if len(diode_times) == n_trials or len(diode_times) == n_trials * 2:
                            fig = plt.figure()
                            plt.plot(times, diode[0, :], 'b')
                            plt.plot([times[0], times[-1]], [diode_thresh, diode_thresh], 'r')
                            for diode_time in diode_times:
                                plt.plot([times[diode_time], times[diode_time]], [np.min(diode, axis=1), diode_thresh],
                                         'g')
                            plt.savefig(
                                op.join(qc_folder,
                                        "{}-{}-{}-diode_{}.png".format(subject_id, session_id, numero, diode_ch_name))
                            )
                            plt.close("all")
                            break
                        else:
                            diode_times = []
                    if len(diode_times) == n_trials or len(diode_times) == 2 * n_trials:
                        break

            # Num of diode onsets
            num_diode_onsets = len(diode_times)
            print('num diode onsets=%d' % num_diode_onsets)

            # In early sessions there was only a diode signal for the dots
            dots_onset = diode_times
            # In later sessions there was a diode for the dots and the instruction stimulus
            if len(diode_times) == n_trials * 2:
                dots_onset = diode_times[0::2]
                instr_onset = diode_times[1::2]

            # Correct event times
            trial_idx = 0
            for i in range(raw_events.shape[0]):
                # Correct dots onset
                if raw_events[i, 2] == 30:
                    # Use photodiode signal if exists
                    if len(diode_times) == n_trials or len(diode_times) == n_trials * 2:
                        raw_events[i, 0] = dots_onset[trial_idx]
                    # Use mean delay if not
                    else:
                        raw_events[i, 0] = raw_events[i, 0] + int(0.0301856 * raw.info['sfreq'])
                # Correct instruction onset
                elif raw_events[i, 2] == 50:
                    # Use photodiode signal if exists
                    if len(diode_times) == n_trials * 2:
                        raw_events[i, 0] = instr_onset[trial_idx]
                    # Use mean delay if not
                    else:
                        # Mean delay is different for first trial
                        if trial_idx == 0:
                            raw_events[i, 0] = raw_events[i, 0] + int(0.0190278 * raw.info['sfreq'])
                        else:
                            raw_events[i, 0] = raw_events[i, 0] + int(0.0302645 * raw.info['sfreq'])
                    trial_idx = trial_idx + 1

            # Downsample data and events together
            raw, events = raw.copy().load_data().resample(
                sfreq,
                npad="auto",
                events=raw_events,
                n_jobs=30,
            )

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
                events,
                overwrite=True
            )

            print("EVENTS SAVED:", eve_path)


if __name__=='__main__':
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
