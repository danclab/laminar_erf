import json
import sys

from meegkit.dss import dss_line
from scipy.signal import welch

from utilities import files
import os.path as op
import mne
import numpy as np
import matplotlib.pylab as plt


def dss_line_iter(data, fline, sfreq, win_sz=10, spot_sz=2.5,
                  nfft=512, show=False, prefix="dss_iter", n_iter_max=100):
    """Remove power line artifact iteratively.

    This method applies dss_line() until the artifact has been smoothed out
    from the spectrum.

    Parameters
    ----------
    data : data, shape=(n_samples, n_chans, n_trials)
        Input data.
    fline : float
        Line frequency.
    sfreq : float
        Sampling frequency.
    win_sz : float
        Half of the width of the window around the target frequency used to fit
        the polynomial (default=10).
    spot_sz : float
        Half of the width of the window around the target frequency used to
        remove the peak and interpolate (default=2.5).
    nfft : int
        FFT size for the internal PSD calculation (default=512).
    show: bool
        Produce a visual output of each iteration (default=False).
    prefix : str
        Path and first part of the visualisation output file
        "{prefix}_{iteration number}.png" (default="dss_iter").
    n_iter_max : int
        Maximum number of iterations (default=100).

    Returns
    -------
    data : array, shape=(n_samples, n_chans, n_trials)
        Denoised data.
    iterations : int
        Number of iterations.
    """

    def nan_basic_interp(array):
        """Nan interpolation."""
        nans, ix = np.isnan(array), lambda x: x.nonzero()[0]
        array[nans] = np.interp(ix(nans), ix(~nans), array[~nans])
        return array

    freq_rn = [fline - win_sz, fline + win_sz]
    freq_sp = [fline - spot_sz, fline + spot_sz]
    freq, psd = welch(data, fs=sfreq, nfft=nfft, axis=0)

    freq_rn_ix = np.logical_and(freq >= freq_rn[0], freq <= freq_rn[1])
    freq_used = freq[freq_rn_ix]
    freq_sp_ix = np.logical_and(freq_used >= freq_sp[0],
                                freq_used <= freq_sp[1])

    if psd.ndim == 3:
        mean_psd = np.mean(psd, axis=(1, 2))[freq_rn_ix]
    elif psd.ndim == 2:
        mean_psd = np.mean(psd, axis=(1))[freq_rn_ix]

    mean_psd_wospot = mean_psd.copy()
    mean_psd_wospot[freq_sp_ix] = np.nan
    mean_psd_tf = nan_basic_interp(mean_psd_wospot)
    pf = np.polyfit(freq_used, mean_psd_tf, 3)
    p = np.poly1d(pf)
    clean_fit_line = p(freq_used)

    aggr_resid = []
    iterations = 0
    while iterations < n_iter_max:
        data, _ = dss_line(data, fline, sfreq, nfft=nfft, nremove=1)
        freq, psd = welch(data, fs=sfreq, nfft=nfft, axis=0)
        if psd.ndim == 3:
            mean_psd = np.mean(psd, axis=(1, 2))[freq_rn_ix]
        elif psd.ndim == 2:
            mean_psd = np.mean(psd, axis=(1))[freq_rn_ix]

        if iterations % 10 == 0:
            mean_psd_wospot = mean_psd.copy()
            mean_psd_wospot[freq_sp_ix] = np.nan
            mean_psd_tf = nan_basic_interp(mean_psd_wospot)
            pf = np.polyfit(freq_used, mean_psd_tf, 3)
            p = np.poly1d(pf)
            clean_fit_line = p(freq_used)

        residuals = mean_psd - clean_fit_line
        mean_score = np.mean(residuals[freq_sp_ix])
        aggr_resid.append(mean_score)

        print("Iteration {} score: {}".format(iterations, mean_score))

        if show:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(2, 2, figsize=(12, 6), facecolor="white")

            if psd.ndim == 3:
                mean_sens = np.mean(psd, axis=2)
            elif psd.ndim == 2:
                mean_sens = psd

            y = mean_sens[freq_rn_ix]
            ax.flat[0].plot(freq_used, y)
            ax.flat[0].set_title("Mean PSD across trials")

            ax.flat[1].plot(freq_used, mean_psd_tf, c="gray")
            ax.flat[1].plot(freq_used, mean_psd, c="blue")
            ax.flat[1].plot(freq_used, clean_fit_line, c="red")
            ax.flat[1].set_title("Mean PSD across trials and sensors")

            tf_ix = np.where(freq_used <= fline)[0][-1]
            ax.flat[2].plot(residuals, freq_used)
            color = "green"
            if mean_score <= 0:
                color = "red"
            ax.flat[2].scatter(residuals[tf_ix], freq_used[tf_ix], c=color)
            ax.flat[2].set_title("Residuals")

            ax.flat[3].plot(np.arange(iterations + 1), aggr_resid, marker='o')
            ax.flat[3].set_title("Iterations")

            f.set_tight_layout(True)
            plt.savefig(f"{prefix}_{iterations:03}.png")
            plt.close("all")

        if mean_score <= 0:
            break

        iterations += 1

    if iterations == n_iter_max:
        raise RuntimeError('Could not converge. Consider increasing the '
                           'maximum number of iterations')

    return data, iterations


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

    meg_path = op.join(session, "meg")

    sess_path = op.join(sub_path, session_id)
    files.make_folder(sess_path)

    qc_folder = op.join(sess_path, "QC")
    files.make_folder(qc_folder)

    raw_paths = files.get_files(sess_path, subject_id+"-"+session_id, "-raw.fif")[2]
    raw_paths.sort()
    event_paths = files.get_files(sess_path, subject_id+"-"+session_id, "-eve.fif")[2]
    event_paths.sort()

    raw_eve_list = list(zip(raw_paths, event_paths))

    # for raw_path, eve_path in [raw_eve_list[0]]:
    for raw_path, eve_path in raw_eve_list:
        print("INPUT RAW FILE:", raw_path)
        print("EVE_RAW MATCH:", raw_path.split("-")[-2] == eve_path.split("-")[-2])
        numero = str(raw_path.split("-")[-2]).zfill(3)

        raw = mne.io.read_raw_fif(raw_path, verbose=False)
        # raw = raw.apply_gradient_compensation(2, verbose=True)
        raw = raw.pick_types(meg=True, eeg=False, ref_meg=True)
        fig = raw.plot_psd(
            tmax=np.inf, fmax=260, average=True, show=False, picks="meg"
        )
        fig.suptitle("{}-{}-{}".format(subject_id, session_id, numero))
        plt.savefig(
            op.join(qc_folder, "{}-{}-{}-raw-psd.png".format(subject_id, session_id, numero)),
            dpi=150, bbox_inches="tight"
        )
        plt.close("all")

        info = raw.info
        raw = raw.get_data()

        zapped, iterations = dss_line_iter(
            raw.transpose(),
            50.0,
            info['sfreq'],
            win_sz=20,
            spot_sz=5.5,
            n_iter_max=500,
            show=True,
            prefix="{}/{}-{}-{}-50_1_iter".format(qc_folder, subject_id, session_id, numero)
        )

        raw = mne.io.RawArray(
            zapped.transpose(),
            info
        )

        fig = raw.plot_psd(tmax=np.inf, fmax=260, average=True, show=False)
        fig.suptitle("{}-{}-{}".format(subject_id, session_id, numero))
        plt.savefig(
            op.join(qc_folder, "{}-{}-{}-zapline-raw-psd.png".format(subject_id, session_id, numero)),
            dpi=150,
            bbox_inches="tight"
        )
        plt.close("all")

        f_n = str(numero).zfill(3)  # file number
        out_path = op.join(
            sess_path,
            "zapline-{}-{}-{}-raw.fif".format(subject_id, session_id, f_n)
        )

        raw.save(
            out_path,
            overwrite=True
        )
        del raw