function out_file=invert_tc(dataset_path, subj_id, session_id, run_id, epo, prior, localizer)

addpath('/home/bonaiuto/spm12')
spm('defaults','eeg');
spm_jobman('initcfg');

patch_size=5;
win_size=10;
win_overlap=true;
n_temp_modes=4;

subj_info=tdfread(fullfile(dataset_path,'raw/participants.tsv'));
s_idx=find(strcmp(cellstr(subj_info.subj_id),subj_id));
nas=subj_info.nas(s_idx,:);
lpa=subj_info.lpa(s_idx,:);
rpa=subj_info.rpa(s_idx,:);

% Where to put output data
data_dir=fullfile(dataset_path,'derivatives/processed',subj_id, session_id);
output_dir=fullfile(data_dir, 'spm');
if exist(output_dir,'dir')~=7
    mkdir(output_dir);
end
subj_fs_dir=fullfile(dataset_path,'derivatives/processed',subj_id,'fs');
subj_surf_dir=fullfile(subj_fs_dir,'surf');

data_file=fullfile(data_dir, sprintf('mspm_converted_autoreject-%s-%s-%s-%s-epo.mat', subj_id, session_id, run_id, epo));
D=spm_eeg_load(data_file);
[path,base,ext]=fileparts(data_file);

mri_fname=fullfile(dataset_path,'raw', subj_id, 'mri', 'headcast/t1w.nii');

% Get times and zero time
times=D.time-D.time(1);

% Meshes to use
pial_fname=fullfile(subj_surf_dir,'pial.ds.link_vector.nodeep.gii');
pial_surf=gifti(pial_fname);
wm_fname=fullfile(subj_surf_dir,'white.ds.link_vector.nodeep.gii');

% Setup mesh lists
mesh_fnames={wm_fname, pial_fname};
mesh_names={'white','pial'};

% Run sliding window inversion at each selected vertex
tc_fvals=zeros(length(mesh_names),length(times));
    
% Because of our cool downsampling method, the vertices in the
% white matter and pial surface are cooresponding
priors={prior, prior};
        
% Run sliding window inversion for each mesh
for m_idx=1:length(mesh_names)
    [path,base,ext]=fileparts(data_file);
    coreg_fname=fullfile(output_dir, sprintf('%s_localizer_%s_sliding_window_%s.mat',localizer,mesh_names{m_idx},base));

    [f_vals,wois]=invert_sliding_window(priors{m_idx},...
        data_file, coreg_fname, mri_fname, mesh_names{m_idx}, mesh_fnames{m_idx},...
        nas, lpa, rpa, patch_size, n_temp_modes,...
        win_size, win_overlap, true);
    tc_fvals(m_idx,:)=f_vals;
end
% Figure out center of each sliding time window
inner_times=wois(:,1)+.5*(wois(:,2)-wois(:,1));
% If overlapping windows, cut off the edges
if win_overlap
    left_idx=round((win_size-1)/2)+1;
    right_idx=length(times)-round((win_size-1)/2);
else
    left_idx=1;
    right_idx=length(centered_wois);
end

invert_tc_results=[];
invert_tc_results.patch_size=patch_size;
invert_tc_results.win_size=win_size;
invert_tc_results.win_overlap=win_overlap;
invert_tc_results.n_temp_modes=n_temp_modes;
invert_tc_results.times=inner_times;
invert_tc_results.left_idx=left_idx;
invert_tc_results.right_idx=right_idx;
invert_tc_results.tc_fvals=tc_fvals;  
invert_tc_results.f_diff=tc_fvals(2,left_idx:right_idx)-tc_fvals(1,left_idx:right_idx);

out_file=fullfile(output_dir, sprintf('invert_%s_localizer_%s_tc_results.json',localizer,base));

fid = fopen(out_file,'w');
fwrite(fid, jsonencode(invert_tc_results)); 
fclose(fid); 
