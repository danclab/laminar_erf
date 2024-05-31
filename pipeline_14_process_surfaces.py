import json
import os
import os.path as op
import sys
from lameg.surf import postprocess_freesurfer_surfaces
from utilities import files


def run(index, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]

    der_path = op.join(path, "derivatives")
    proc_path = op.join(der_path, "processed")

    subjects = files.get_folders(proc_path, 'sub-', '')[2]
    subjects.sort()
    subject = subjects[index]
    subject_id = subject.split("/")[-1]
    print("ID:", subject_id)

    subj_output_path=os.path.join(proc_path, subject, 'surf')
    if not os.path.exists(subj_output_path):
        os.mkdir(subj_output_path)
        # Create an 11-layer surface
    postprocess_freesurfer_surfaces(
        subject,
        subj_output_path,
        'multilayer.11.ds.link_vector.fixed.gii',
        n_surfaces=11,
        ds_factor=0.1,
        orientation='link_vector',
        remove_deep=True
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