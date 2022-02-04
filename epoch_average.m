function epoch_average(fname)

addpath('/home/bonaiuto/Dropbox/Toolboxes/spm12')
spm('defaults','eeg');
spm_jobman('initcfg');

clear jobs
matlabbatch={};
batch_idx=1;

matlabbatch{batch_idx}.spm.meeg.averaging.average.D = {fname};
matlabbatch{batch_idx}.spm.meeg.averaging.average.userobust.robust.ks = 3;
matlabbatch{batch_idx}.spm.meeg.averaging.average.userobust.robust.bycondition = false;
matlabbatch{batch_idx}.spm.meeg.averaging.average.userobust.robust.savew = false;
matlabbatch{batch_idx}.spm.meeg.averaging.average.userobust.robust.removebad = false;
matlabbatch{batch_idx}.spm.meeg.averaging.average.plv = false;
matlabbatch{batch_idx}.spm.meeg.averaging.average.prefix = 'm';
batch_idx=batch_idx+1;
spm_jobman('run', matlabbatch);    
