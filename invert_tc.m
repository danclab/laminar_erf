function invert_tc(dataset_path, subj_id, session_id, run_id, epo)

patch_size=5;
win_size=10;
win_overlap=true;
n_temp_modes=4;

spm('defaults','eeg');
spm_jobman('initcfg');

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

data_file=fullfile(data_dir, sprintf('mspm_converted_autoreject-%s-%s-%s-%s-epo.mat', subj_id, session_id, run_id, epo));
D=spm_eeg_load(data_file);
[path,base,ext]=fileparts(data_file);
load(fullfile(output_dir, sprintf('invert_%s_localizer_results.mat',base)));
pial_clusters=invert_localizer_results.clusters;

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

% For each cluster found by the localizer
for cluster_idx=1:length(pial_clusters)
    cluster=pial_clusters(cluster_idx);
    max_vert=cluster.vertices(cluster.max_idx);
    
    % Difference in link vector angle at max vertex of cluster and all
    % other vertices in cluster
    norm_diffs=[];
    for i=1:length(cluster.vertices)
        norm_diffs(i)=atan2(norm(cross(pial_surf.normals(max_vert,:),pial_surf.normals(cluster.vertices(i),:))),...
            dot(pial_surf.normals(max_vert,:),pial_surf.normals(cluster.vertices(i),:)));
    end
    % Invert at each vertex with link vector angle within .1 radians of the
    % max vertex
    cluster.inv_verts=cluster.vertices(find(abs(norm_diffs<.1)));

    % Run sliding window inversion at each selected vertex
    cluster.tc_fvals=zeros(length(cluster.inv_verts),length(mesh_names),length(times));
    for v_idx=1:length(cluster.inv_verts)
        pial_prior=cluster.inv_verts(v_idx);

        % Because of our cool downsampling method, the vertices in the
        % white matter and pial surface are cooresponding
        priors={pial_prior, pial_prior};
        % Only have to recompute the lead field gain matrix once
        recompute_lgain=(cluster_idx==1) & (v_idx==1);
        
        % Run sliding window inversion for each mesh
        for m_idx=1:length(mesh_names)
            [path,base,ext]=fileparts(data_file);
            coreg_fname=fullfile(output_dir, sprintf('%s_sliding_window_%d_%d_%s.mat',mesh_names{m_idx},cluster_idx,v_idx,base));

            [cluster.tc_fvals(v_idx,m_idx,:),wois]=invert_sliding_window(priors{m_idx},...
                data_file, coreg_fname, mri_fname, mesh_names{m_idx}, mesh_fnames{m_idx},...
                nas, lpa, rpa, patch_size, n_temp_modes,...
                win_size, win_overlap, recompute_lgain);
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
    end

    % Compute free energy difference
    cluster.f_diff=squeeze(cluster.tc_fvals(:,2,left_idx:right_idx)-cluster.tc_fvals(:,1,left_idx:right_idx));
    cluster.f_diff=reshape(cluster.f_diff,[size(cluster.tc_fvals,1) (right_idx-left_idx+1)]); 
    new_pial_clusters(cluster_idx)=cluster;
end

invert_tc_results=[];
invert_tc_results.patch_size=patch_size;
invert_tc_results.win_size=win_size;
invert_tc_results.win_overlap=win_overlap;
invert_tc_results.n_temp_modes=n_temp_modes;
invert_tc_results.times=inner_times;
invert_tc_results.left_idx=left_idx;
invert_tc_results.right_idx=right_idx;
invert_tc_results.clusters=new_pial_clusters;   

save(fullfile(output_dir, sprintf('invert_%s_tc_results.mat',base)), 'invert_tc_results');

fig=figure();
for cluster_idx=1:length(new_pial_clusters)
    subplot(length(new_pial_clusters),1,cluster_idx);
    hold all
    plot(inner_times(left_idx:right_idx),new_pial_clusters(cluster_idx).f_diff');
end
plot([inner_times(left_idx) inner_times(right_idx)],[0 0],'k');
plot([inner_times(left_idx) inner_times(right_idx)],[3 3],'k--');
plot([inner_times(left_idx) inner_times(right_idx)],[-3 -3],'k--');
xlim([inner_times(left_idx) inner_times(right_idx)]);
xlabel('Time (ms)')
ylabel('Fpial-Fwhite');
