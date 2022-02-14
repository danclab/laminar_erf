function epoch_average(fname)

addpath('/home/bonaiuto/spm12');
spm('defaults','eeg');
spm_jobman('initcfg');

clear jobs
matlabbatch={};

matlabbatch{1}.spm.meeg.averaging.average.D = {fname};
matlabbatch{1}.spm.meeg.averaging.average.userobust.standard = false;
matlabbatch{1}.spm.meeg.averaging.average.plv = false;
matlabbatch{1}.spm.meeg.averaging.average.prefix = 'm';

spm_jobman('run', matlabbatch);    
