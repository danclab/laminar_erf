function epoch_average(dataset_path, subj_id, session_id, run_id, epo)

spm('defaults','eeg');
spm_jobman('initcfg');

data_path=fullfile(dataset_path, 'derivatives/processed', subj_id, session_id);
fname=fullfile(data_path, sprintf('spm_converted_autoreject-%s-%s-%s-%s-epo.mat', subj_id, session_id, run_id, epo));
clear jobs
matlabbatch={};
batch_idx=1;

matlabbatch{batch_idx}.spm.meeg.averaging.average.D = {fname};
matlabbatch{batch_idx}.spm.meeg.averaging.average.userobust.standard = false;
matlabbatch{batch_idx}.spm.meeg.averaging.average.plv = false;
matlabbatch{batch_idx}.spm.meeg.averaging.average.prefix = 'm';
batch_idx=batch_idx+1;
spm_jobman('run', matlabbatch);    
