function fig=plot_mask(subj_surf_dir, surface, mask)

% Subject surfaces
pial_ds_fname=fullfile(subj_surf_dir, 'pial.ds.gii');
pial_ds=gifti(pial_ds_fname);    
pial_inflated_fname=fullfile(subj_surf_dir, 'pial.ds.inflated.gii');
pial_inflated=gifti(pial_inflated_fname);    
mapping=knnsearch(pial_ds.vertices,surface.vertices);

data=zeros(size(pial_inflated.vertices,1),1);
data(mapping(mask))=1;                

fig=figure();
ax=subplot(1,1,1);
[ax,metric_data]=plot_surface_metric(pial_inflated, data, 'ax', ax,...
    'clip_vals',false, 'threshold', 0.1, 'limits', [0 1], 'custom_cm', false,...
    'specular_strength', 0.0, 'ambient_strength', 0.8, 'face_lighting', '');
set(ax,'CameraViewAngle',6.028);
