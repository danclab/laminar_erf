function mask=create_mask(subj_info, hemispheres, regions, pial_ds_fname, varargin)

subj_surf_dir=fullfile(params.surf_dir,sprintf('%s-synth',subj_info.subj_id),'surf');

mask=[];

pial_original_fname=fullfile(subj_surf_dir,'pial.gii');
pial_orig=gifti(pial_original_fname);
pial_ds=gifti(pial_ds_fname);

mapping=knnsearch(pial_orig.vertices,pial_ds.vertices);

hemisphere_map=get_hemisphere_map(pial_ds_fname, pial_original_fname, 'recompute', false);
[lh_vertices, lh_label, lh_colortable] = read_annotation(fullfile(params.surf_dir,sprintf('%s-synth',subj_info.subj_id),'label', 'lh.aparc.annot'));
[rh_vertices, rh_label, rh_colortable] = read_annotation(fullfile(params.surf_dir,sprintf('%s-synth',subj_info.subj_id),'label', 'rh.aparc.annot'));

if find(strcmp(hemispheres,'lh'))
    for i=1:size(pial_ds.vertices,1)
        orig_vtx=mapping(i);        
        if hemisphere_map(i)==1
            if lh_label(orig_vtx)>0
                struct_idx=find(lh_colortable.table(:,5)==lh_label(orig_vtx));
                region=lh_colortable.struct_names{struct_idx};
                if find(strcmp(regions,region))
                    mask(end+1)=i;
                end
            end
        end
    end
end
if find(strcmp(hemispheres,'rh'))
    for i=1:size(pial_ds.vertices,1)
        orig_vtx=mapping(i)-length(lh_vertices);        
        if hemisphere_map(i)==2
            if rh_label(orig_vtx)>0
                struct_idx=find(rh_colortable.table(:,5)==rh_label(orig_vtx));
                region=rh_colortable.struct_names{struct_idx};
                if find(strcmp(regions,region))
                    mask(end+1)=i;
                end
            end
        end
    end
end
