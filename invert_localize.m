function out_file=invert_localize(dataset_path, subj_id, session_id, run_id, epo)

addpath('/home/bonaiuto/spm12')
addpath('/home/bonaiuto/MEGsurfer');
spm('defaults','eeg');
spm_jobman('initcfg');

set(0,'DefaultFigureVisible','off')

patch_size=5;
n_temp_modes=4;
woi=[-Inf Inf];
if contains(epo, 'motor')
    woi=[-100 200];
elseif contains(epo,'visual1')
    woi=[100 300];
elseif contains(epo,'visual2')
    woi=[100 300];
end

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

% Data file to load
data_file=fullfile(data_dir, sprintf('fmspm_converted_autoreject-%s-%s-%s-%s-epo.mat', subj_id, session_id, run_id, epo));

mri_fname=fullfile(dataset_path,'raw', subj_id, 'mri', 'headcast/t1w.nii');
    
surfaces={'pial','white'};

invert_localizer_results=[];
invert_localizer_results.subj_id=subj_id;
invert_localizer_results.session_id=session_id;
invert_localizer_results.run_id=run_id;
invert_localizer_results.epo=epo;
invert_localizer_results.patch_size=patch_size;
invert_localizer_results.n_temp_modes=n_temp_modes;
invert_localizer_results.surfaces=surfaces;
invert_localizer_results.surf_fname={};
invert_localizer_results.F=[];
invert_localizer_results.res_surf_fname={};
invert_localizer_results.mu_fname={};
invert_localizer_results.it_fname={};
invert_localizer_results.res_fname={};

for s_idx=1:length(surfaces)
    surface=surfaces{s_idx};
    
    surf_fname=fullfile(subj_surf_dir,sprintf('%s.ds.link_vector.nodeep.gii', surface));

    % Subject surfaces
    
    invert_localizer_results.surf_fname{s_idx}=surf_fname;
    
    % Create smoothed meshes
    [smoothkern]=spm_eeg_smoothmesh_mm(surf_fname, patch_size);

    % Coregistered filename
    [path,base,ext]=fileparts(data_file);
    coreg_fname=fullfile(output_dir, sprintf('%s_localize_%s.mat',surface,base));
 
    % Run inversion
    out_file=invert_ebb(data_file, coreg_fname, mri_fname, surf_fname,...
        nas, lpa, rpa, patch_size, n_temp_modes, woi);
    invert_localizer_results.res_surf_fname{s_idx}=out_file;

    % Load mesh results
    mesh_results=gifti(out_file);
    D=spm_eeg_load(coreg_fname);
    M=D.inv{1}.inverse.M;
    U=D.inv{1}.inverse.U{1};
    MU=M*U;
    It   = D.inv{1}.inverse.It;

    mu_fname=fullfile(output_dir, sprintf('%s_localize_MU_%s.tsv',surface,base));
    dlmwrite(mu_fname, MU, '\t');
    invert_localizer_results.mu_fname{s_idx}=mu_fname;

    it_fname=fullfile(output_dir, sprintf('%s_localize_It_%s.tsv',surface,base));
    dlmwrite(it_fname, It, '\t');
    invert_localizer_results.it_fname{s_idx}=it_fname;

    res_fname=fullfile(output_dir, sprintf('%s_localize_res_%s.tsv',surface,base));
    dlmwrite(res_fname, mesh_results.cdata(:), '\t');
    invert_localizer_results.res_fname{s_idx}=res_fname;
    
    invert_localizer_results.F(s_idx)=D.inv{1}.inverse.crossF;
end
    
out_file=fullfile(output_dir, sprintf('invert_%s_localizer_results.json',base));

fid = fopen(out_file,'w');
fwrite(fid, jsonencode(invert_localizer_results)); 
fclose(fid); 



