function run_inversions(subj_idx)

addpath('/home/bonaiuto/Dropbox/Toolboxes/spm12')
spm('defaults','eeg');

fname = 'settings.json'; 
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
settings = jsondecode(str);

processed_dir=fullfile(settings.dataset_path,'derivatives','processed');
subjs=dir(fullfile(processed_dir,'sub*'));
subj_id=subjs(subj_idx).name;

subj_dir=fullfile(processed_dir, subj_id);
sessions=dir(fullfile(subj_dir,'ses*'));

for sess_idx=1:length(sessions)
    sess_id=sessions(sess_idx).name;
    sess_dir=fullfile(subj_dir,sess_id);
    
    spm_files=dir(fullfile(sess_dir,'spm_*.mat'));
    for spm_idx=1:length(spm_files)
        spm_file=spm_files(spm_idx).name;
        pieces=split(spm_file,'-');
        block_id=pieces{6};
        epo=pieces{7};
        epoch_average(settings.dataset_path,subj_id,sess_id,block_id,epo);
        invert_localize(settings.dataset_path,subj_id,sess_id,block_id,epo);
        invert_tc(settings.dataset_path,subj_id,sess_id,block_id,epo);
    end
end