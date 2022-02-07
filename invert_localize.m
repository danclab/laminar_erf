function invert_localize(dataset_path, subj_id, session_id, run_id, epo)

patch_size=5;
n_temp_modes=4;

% Threshold percentage
percent_thresh=.95;

subj_info=tdfread(fullfile(dataset_path,'raw/participants.tsv'));
s_idx=find(strcmp(cellstr(subj_info.subj_id),subj_id));
nas=subj_info.nas(s_idx,:);%cellfun(@str2num, split(subj_info.nas(s_idx,:),','));
lpa=subj_info.lpa(s_idx,:);%cellfun(@str2num, split(subj_info.lpa(s_idx,:),','));
rpa=subj_info.rpa(s_idx,:);%cellfun(@str2num, split(subj_info.rpa(s_idx,:),','));

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
    
% Subject surfaces
% Compute link vectors and save in pial surface
pial_orig_fname=fullfile(subj_surf_dir,'pial.gii');
pial_orig=gifti(pial_orig_fname);
pial_ds_fname=fullfile(subj_surf_dir,'pial.ds.gii');
pial_ds=gifti(pial_ds_fname);
norm=compute_surface_normals(subj_surf_dir, 'pial', 'link_vector');
pial_ds.normals=norm;
pial_ds_lv_fname=fullfile(subj_surf_dir,'pial.ds.link_vector.gii');
save(pial_ds,pial_ds_lv_fname);
pial_ds_final=remove_deep_vertices(subj_fs_dir, pial_ds, pial_orig, pial_ds_lv_fname, pial_orig_fname);
pial_ds_lv_rm_fname=fullfile(subj_surf_dir,'pial.ds.link_vector.nodeep.gii');
save(pial_ds_final,pial_ds_lv_rm_fname);

% Compute link vectors and save in white matter surface
white_orig_fname=fullfile(subj_surf_dir,'white.gii');
white_orig=gifti(white_orig_fname);
white_ds_fname=fullfile(subj_surf_dir,'white.ds.gii');
white_ds=gifti(white_ds_fname);
norm=compute_surface_normals(subj_surf_dir, 'white', 'link_vector');
white_ds.normals=norm;
white_ds_lv_fname=fullfile(subj_surf_dir,'white.ds.link_vector.gii');
save(white_ds,white_ds_lv_fname);
white_ds_final=remove_deep_vertices(subj_fs_dir, white_ds, white_orig, white_ds_lv_fname, white_orig_fname);
white_ds_lv_rm_fname=fullfile(subj_surf_dir,'white.ds.link_vector.nodeep.gii');
save(white_ds_final,white_ds_lv_rm_fname);

spm('defaults','eeg');
spm_jobman('initcfg');

% Create smoothed meshes
[smoothkern]=spm_eeg_smoothmesh_mm(pial_ds_lv_rm_fname, patch_size);

% Coregistered filename
[path,base,ext]=fileparts(data_file);
coreg_fname=fullfile(output_dir, sprintf('pial_localize_%s.mat',base));
 
% Run inversion
inv_results=invert_ebb(data_file, coreg_fname, mri_fname, pial_ds_lv_rm_fname,...
    nas, lpa, rpa, patch_size, n_temp_modes);

% Load mesh results
mesh_results=gifti(inv_results);
D=spm_eeg_load(coreg_fname);
goodchans=D.indchantype('MEG','good');
M=D.inv{1}.inverse.M;
U=D.inv{1}.inverse.U{1};
MU=M*U;
It   = D.inv{1}.inverse.It;
Dgood=squeeze(D(goodchans,It,:));

% Find vertices in mask where results within percentile threshold
%mask=create_mask(subj_info, hemispheres, regions, pial_ds_lv_rm_fname);

%cluster_mask=intersect(mask,find(mesh_results.cdata(:)>=max(mesh_results.cdata(mask))*percent_thresh));
cluster_mask=find(mesh_results.cdata(:)>=max(mesh_results.cdata(:))*percent_thresh);

fig=plot_surface_metric(pial_ds_final,mesh_results.cdata(:),'clip_vals',false);
saveas(fig, fullfile(output_dir, sprintf('pial_localize_%s.png', base)));

fig=plot_mask(subj_surf_dir, pial_ds_final, cluster_mask);
saveas(fig, fullfile(output_dir, sprintf('pial_localize_mask_%s.png', base)));

cluster_mesh=struct();
% Get vertices and all faces containing a vertex from the ROI
cluster_mesh.vertices=pial_ds_final.vertices;
% Find faces with vertices in p_mask
[rows,cols]=find(ismember(pial_ds_final.faces,cluster_mask));
% Only include faces with more than one vertex in mask
cluster_mesh.faces=pial_ds_final.faces(unique(rows),:);
% Split into clusters
cluster_fv=splitFV(cluster_mesh);
    
clusters=[];
for c_idx=1:length(cluster_fv)
    vs=intersect(cluster_mask, find(ismember(pial_ds_final.vertices,cluster_fv(c_idx).vertices,'rows')>0));
    clusters(c_idx).vertices=vs;
    max_idx=find(mesh_results.cdata(vs)==max(mesh_results.cdata(vs)));
    clusters(c_idx).max_idx=max_idx;
    clusters(c_idx).coords=pial_ds_final.vertices(vs,:);
    clusters(c_idx).source_tc=MU(vs(max_idx),:)*squeeze(Dgood(:,:,1));
end    

invert_localizer_results=[];
invert_localizer_results.subj_id=subj_id;
invert_localizer_results.session_id=session_id;
invert_localizer_results.run_id=run_id;
invert_localizer_results.epo=epo;
invert_localizer_results.patch_size=patch_size;
invert_localizer_results.n_temp_modes=n_temp_modes;
invert_localizer_results.percent_thresh=percent_thresh;
invert_localizer_results.data_file=data_file;
invert_localizer_results.clusters=clusters;
invert_localizer_results.times=D.time;
%invert_subject_results.regions=regions;
%invert_subject_results.hemispheres=hemispheres;
save(fullfile(output_dir, sprintf('invert_%s_localizer_results.mat',base)), 'invert_localizer_results');


