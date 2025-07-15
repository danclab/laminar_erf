import csv
import os
import sys
import json
import nibabel as nib
from mne import pick_channels_regexp
import numpy as np
from mne.coreg import fit_matched_points
from mne.io.ctf.trans import _make_ctf_coord_trans_set
from mne.transforms import apply_trans, Transform
import matplotlib.pyplot as plt
from utilities import files
import os.path as op
import mne
from scipy.optimize import least_squares

def get_fiducial_coords(subj_id, fname, col_delimiter='\t', subject_column='subj_id',
                        nas_column='nas', lpa_column='lpa', rpa_column='rpa', val_delimiter=','):
    """
    Fetch fiducial coordinates from a tab-separated values (TSV) file for a given subject ID.

    Parameters
    ----------
    subj_id : str
        The subject ID to search for in the TSV file.
    fname : str
        Path to the TSV file.
    col_delimiter : str, optional
        Column delimiter when reading file. Default is \t.
    subject_column : str, optional
        Column name for subject. Default is subj_id.
    nas_column : str, optional
        Column name for nas coordinate. Default is nas.
    lpa_column : str, optional
        Column name for lpa coordinate. Default is lpa.
    rpa_column : str, optional
        Column name for rpa coordinate. Default is rpa.
    val_delimiter : str, optional
        Value delimiter when reading file. Default is ,.

    Returns
    -------
    NAS : list
        List of floats representing the NASion fiducial coordinates.
    LPA : list
        List of floats representing the Left Preauricular fiducial coordinates.
    RPA : list
        List of floats representing the Right Preauricular fiducial coordinates.
    """

    with open(fname, 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=col_delimiter)
        for row in reader:
            if row[subject_column] == subj_id:
                nas = [float(i)/1000 for i in row[nas_column].split(val_delimiter)]
                lpa = [float(i)/1000 for i in row[lpa_column].split(val_delimiter)]
                rpa = [float(i)/1000 for i in row[rpa_column].split(val_delimiter)]
                return nas, lpa, rpa

    return None, None, None  # Return None for each if no matching subj_id is found

def sphere_from_fiducials(nas, lpa, rpa):
    A = np.vstack([nas, lpa, rpa])
    b = np.sum(A ** 2, axis=1)

    # Solve linear system to find sphere center
    A = np.hstack((2 * A, np.ones((3, 1))))
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    center = params[:3]
    return center

def compute_origin_from_meg_fids(raw):
    fids = []
    for ident in (1, 2, 3):  # NAS=1, LPA=2, RPA=3
        for d in raw.info['dig']:
            if d['kind'] == 1 and d['ident'] == ident:
                fids.append(d['r'])
    nas_head, lpa_head, rpa_head = fids
    origin_head = sphere_from_fiducials(nas_head, lpa_head, rpa_head)
    return origin_head

# def compute_trans_from_fiducials(raw, mri_fids):
#     """Compute MRI?head transform from fiducials."""
#     meg_fids = []
#     for ident in (1, 2, 3):  # NAS=1, LPA=2, RPA=3
#         for d in raw.info['dig']:
#             if d['kind'] == 1 and d['ident'] == ident:  # 1 = FIFFV_POINT_CARDINAL
#                 meg_fids.append(d['r'])
#     meg_fids = np.array(meg_fids)
#     trans_mat = fit_matched_points(meg_fids, mri_fids)
#     return Transform('mri', 'head', trans_mat)


# def fit_sphere(coords):
#     # Sphere fitting objective function
#     def fun(center):
#         radii = np.linalg.norm(coords - center, axis=1)
#         return radii - np.mean(radii)
#
#     # Initial guess (mean of points)
#     x0 = np.mean(coords, axis=0)
#     res = least_squares(fun, x0)
#     return res.x
#
# def get_individual_sphere_origin(subject_id, subjects_dir, trans):
#     bem_path = op.join(subjects_dir, subject_id, 'bem', f'{subject_id}-head.fif')
#     bem_surfs = mne.read_bem_surfaces(bem_path)
#     scalp = [s for s in bem_surfs if s['id'] == 4][0]  # outer_skin
#
#     # Fit a sphere robustly
#     scalp_center_mri = fit_sphere(scalp['rr'])
#
#     # Transform to head coordinates
#     origin_head = apply_trans(trans['trans'], scalp_center_mri)
#
#     return origin_head

# def get_individual_sphere_origin(subject_id, subjects_dir, trans):
#     """Estimate individualized origin for SSS from BEM scalp surface."""
#     bem_path = op.join(subjects_dir, subject_id, 'bem', f'{subject_id}-head.fif')
#     bem_surfs = mne.read_bem_surfaces(bem_path)
#     scalp = [s for s in bem_surfs if s['id'] == 4][0]  # outer_skin
#     scalp_center_mri = np.mean(scalp['rr'], axis=0)
#     origin_head = apply_trans(trans['trans'], scalp_center_mri)
#     return origin_head  # already in meters


def run(index, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)
    path = parameters["dataset_path"]
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

    # subjects_file='/home/common/bonaiuto/cued_action_meg/raw/participants.tsv'
    # nas, lpa, rpa = get_fiducial_coords(subject_id, subjects_file)
    # # mri_fiducials=[nas, lpa, rpa]
    # print(nas)
    # print(lpa)
    # print(rpa)
    # subjects_dir = '/home/common/bonaiuto/cued_action_meg/derivatives/processed/fs/'

    # # Load MRI scan
    # mri_img = nib.load(op.join(subjects_dir, subject_id, 'mri', 'T1.mgz'))
    #
    # # Your current MRI fiducials (scanner coordinates in meters)
    # scanner_fids = np.array([nas, lpa, rpa]) * 1000  # convert back to mm temporarily
    #
    # # Convert to voxel indices (scanner space)
    # vox_fids = nib.affines.apply_affine(np.linalg.inv(mri_img.header.get_vox2ras()), scanner_fids)
    #
    # # Convert voxel indices ? FreeSurfer RAS
    # fs_ras_fids = nib.affines.apply_affine(mri_img.header.get_vox2ras_tkr(), vox_fids)
    #
    # # Finally, convert to meters for MNE
    # fs_ras_fids_m = fs_ras_fids / 1000
    #
    # print("FS RAS fiducials (m):", fs_ras_fids_m)
    #
    # if not os.path.exists(os.path.join(subjects_dir,subject_id,'bem/watershed')):
    #     mne.bem.make_watershed_bem(
    #         subject=subject_id,
    #         subjects_dir=subjects_dir,
    #         overwrite=True
    #     )
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

        dss = files.get_folders_files(meg_path)[0]
        dss = [i for i in dss if "ds" in i]
        dss.sort()

        for ds in dss:
            print("INPUT RAW FILE:", ds)
            numero = int(ds.split(".")[0][-2:])
            f_n = str(numero).zfill(3)  # file number

            raw = mne.io.read_raw_ctf(
                ds,
                clean_names=True,
                verbose=False
            )

            raw_events = mne.find_events(
                raw,
                stim_channel="UPPT001",
                min_duration=0.002,
                verbose="DEBUG",
                consecutive=True
            )

            # Get indices of relevant events
            all_evts = np.where(
                np.isin(raw_events[:, 2], [10, 20, 30, 40, 50, 60, 70])
            )[0]

            # Find time of last relevant event
            last_sample = raw_events[all_evts[-1], 0]
            sfreq = raw.info['sfreq']
            crop_tmax = (last_sample / sfreq) + 1.0

            # Crop raw data
            if crop_tmax<raw.times[-1]:
                raw.crop(tmax=crop_tmax)

            # Pick channels corresponding to the cHPI positions
            hpi_picks = pick_channels_regexp(raw.info['ch_names'], 'HLC00[123][123].*')

            # make sure we get 9 channels
            if len(hpi_picks) != 9:
                raise RuntimeError('Could not find all 9 cHPI channels')

            # get indices in alphabetical order
            sorted_picks = np.array(sorted(hpi_picks,
                                           key=lambda k: raw.info['ch_names'][k]))

            # make picks to match order of dig cardinial ident codes.
            # LPA (HPIC002[123]-*), NAS(HPIC001[123]-*), RPA(HPIC003[123]-*)
            hpi_picks = sorted_picks[[3, 4, 5, 0, 1, 2, 6, 7, 8]]
            del sorted_picks

            # process the entire run
            time_sl = slice(0, len(raw.times))
            chpi_data = raw[hpi_picks, time_sl][0]

            # transforms
            tmp_trans = _make_ctf_coord_trans_set(None, None)
            ctf_dev_dev_t = tmp_trans['t_ctf_dev_dev']
            del tmp_trans

            # find indices where chpi locations change (threshold is 0.00001)
            indices = [0]
            indices.extend(np.where(np.any(np.abs(np.diff(chpi_data, axis=1))>0.00001,axis=0))[0]+ 1)
            # data in channels are in ctf device coordinates (cm)
            rrs = chpi_data[:, indices].T.reshape(len(indices), 3, 3)  # m
            # map to mne device coords
            rrs = apply_trans(ctf_dev_dev_t, rrs)
            gofs = np.ones(rrs.shape[:2])  # not encoded, set all good
            moments = np.zeros(rrs.shape)  # not encoded, set all zero
            times = raw.times[indices] + raw._first_time
            chpi_locs = dict(rrs=rrs, gofs=gofs, times=times, moments=moments)

            head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=False)

            used_coils = np.array([0, 1, 2])
            coil_labels = ['lpa', 'nas', 'rpa']

            plt.figure()

            plt.subplot(3, 1, 1)
            for idx, i in enumerate(used_coils):
                c = chpi_locs['rrs'][:, i, 0] - np.mean(chpi_locs['rrs'][:, i, 0])
                plt.plot(chpi_locs['times'], c * 1000, label=coil_labels[idx])
            plt.legend()
            # plt.ylim([-10,10])
            plt.xlim(chpi_locs['times'][[0, -1]])
            plt.ylabel('x (mm)')

            plt.subplot(3, 1, 2)
            for idx, i in enumerate(used_coils):
                c = chpi_locs['rrs'][:, i, 1] - np.mean(chpi_locs['rrs'][:, i, 1])
                plt.plot(chpi_locs['times'], c * 1000)
            # plt.ylim([-15,15])
            plt.xlim(chpi_locs['times'][[0, -1]])
            plt.ylabel('y (mm)')

            plt.subplot(3, 1, 3)
            for idx, i in enumerate(used_coils):
                c = chpi_locs['rrs'][:, i, 2] - np.mean(chpi_locs['rrs'][:, i, 2])
                plt.plot(chpi_locs['times'], c * 1000)
            # plt.ylim([-15,15])
            plt.xlim(chpi_locs['times'][[0, -1]])
            plt.ylabel('z (mm)')
            plt.xlabel('time (s)')

            plt.savefig(
                op.join(qc_folder, "{}-{}-{}-chpi.png".format(subject_id, session_id, numero)),
                dpi=150, bbox_inches="tight"
            )
            plt.close("all")

            for idx, i in enumerate(used_coils):
                sd = np.std(chpi_locs['rrs'][:, i, 0]) * 1000
                print(f'{coil_labels[idx]}, x SD={sd:.2f} mm')
            for idx, i in enumerate(used_coils):
                sd = np.std(chpi_locs['rrs'][:, i, 1]) * 1000
                print(f'{coil_labels[idx]}, y SD={sd:.2f} mm')
            for idx, i in enumerate(used_coils):
                sd = np.std(chpi_locs['rrs'][:, i, 2]) * 1000
                print(f'{coil_labels[idx]}, z SD={sd:.2f} mm')

            lpa_pos = chpi_locs['rrs'][:, used_coils[0], :]
            nas_pos = chpi_locs['rrs'][:, used_coils[1], :]
            rpa_pos = chpi_locs['rrs'][:, used_coils[2], :]

            lpa_rpa_dist = np.sqrt(np.sum((lpa_pos - rpa_pos) ** 2, axis=-1))
            lpa_nas_dist = np.sqrt(np.sum((lpa_pos - nas_pos) ** 2, axis=-1))
            rpa_nas_dist = np.sqrt(np.sum((rpa_pos - nas_pos) ** 2, axis=-1))

            plt.figure()
            plt.plot(lpa_rpa_dist, label='lpa-rpa')
            plt.plot(lpa_nas_dist, label='lpa-nas')
            plt.plot(rpa_nas_dist, label='rpa-nas')
            plt.legend()
            plt.savefig(
                op.join(qc_folder, "{}-{}-{}-chpi_dists.png".format(subject_id, session_id, numero)),
                dpi=150, bbox_inches="tight"
            )
            plt.close("all")

            print(f'LPA-RPA = {np.mean(lpa_rpa_dist) * 1000} mm')
            print(f'LPA-NAS = {np.mean(lpa_nas_dist) * 1000} mm')
            print(f'RPA-NAS = {np.mean(rpa_nas_dist) * 1000} mm')

            fig = mne.viz.plot_head_positions(head_pos, mode="traces", show=False)
            fig.savefig(
                op.join(qc_folder, "{}-{}-{}-head_pos.png".format(subject_id, session_id, numero)),
                dpi=150, bbox_inches="tight"
            )

            fig = mne.viz.plot_head_positions(
                head_pos, mode="field", destination=raw.info["dev_head_t"], info=raw.info,
                show=False
            )  # visualization 3D
            fig.savefig(
                op.join(qc_folder, "{}-{}-{}-head_pos_3d.png".format(subject_id, session_id, numero)),
                dpi=150, bbox_inches="tight"
            )

            # # Compute MRI?head transform from fiducials
            # trans = compute_trans_from_fiducials(raw, fs_ras_fids_m)
            # mne.write_trans(op.join(subjects_dir, subject_id, 'bem', f'{subject_id}-{session_id}-{numero}-trans.fif'), trans, overwrite=True)
            #
            # mne.viz.plot_alignment(raw.info, trans=trans, subject=subject_id,
            #                        subjects_dir=subjects_dir)#, surfaces=['head-dense'])
            #
            # # Estimate individualized origin
            # origin = get_individual_sphere_origin(subject_id, subjects_dir, trans)
            # origin = compute_origin_from_meg_fids(raw)
            # print(f"Using individualized SSS origin (head coords): {origin}")

            raw_sss = mne.preprocessing.maxwell_filter(
                raw, head_pos=head_pos,
                st_duration=10,
                origin=[0., 0., 0.04],
                coord_frame='head',
                verbose=True
            )

            raw_path = op.join(
                sess_path,
                "{}-{}-{}-raw.fif".format(subject_id, session_id, f_n)
            )
            raw_sss.save(
                raw_path,
                fmt="single",
                overwrite=True)

            print("RAW SAVED:", raw_path)

            raw.close()
            raw_sss.close()

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