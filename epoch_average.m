function epoch_average(fname)

addpath('/home/bonaiuto/spm12');
spm('defaults','eeg');
spm_jobman('initcfg');

clear jobs
matlabbatch={};

matlabbatch{1}.spm.meeg.averaging.average.D = {fname};
matlabbatch{1}.spm.meeg.averaging.average.userobust.robust.ks = 3;
matlabbatch{1}.spm.meeg.averaging.average.userobust.robust.bycondition = false;
matlabbatch{1}.spm.meeg.averaging.average.userobust.robust.savew = false;
matlabbatch{1}.spm.meeg.averaging.average.userobust.robust.removebad = false;
matlabbatch{1}.spm.meeg.averaging.average.plv = false;
matlabbatch{1}.spm.meeg.averaging.average.prefix = 'm';
matlabbatch{2}.spm.meeg.preproc.filter.D(1) = cfg_dep('Averaging: Averaged Datafile', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{2}.spm.meeg.preproc.filter.type = 'butterworth';
matlabbatch{2}.spm.meeg.preproc.filter.band = 'low';
matlabbatch{2}.spm.meeg.preproc.filter.freq = 120;
matlabbatch{2}.spm.meeg.preproc.filter.dir = 'twopass';
matlabbatch{2}.spm.meeg.preproc.filter.order = 5;
matlabbatch{2}.spm.meeg.preproc.filter.prefix = 'f';


spm_jobman('run', matlabbatch);    
