json_file = "settings.json";
parameters = read_json_file(json_file);

base_dir=fullfile(parameters.dataset_path, 'derivatives/processed');

spm('defaults','eeg');

subj_dirs=dir(base_dir);
for 3:len(sub_dirs)
    subject=subj_dirs(s).name;
    if length(strfind(subject,'sub'))
        subject_dir=fullfile(base_dir, subject);
        ses_dirs=dir(subject_dir);
        for se=3:length(ses_dirs)
            session=ses_dirs(se).name;
            if length(strfind(session,'ses'))
                session_dir=fullfile(subject_dir, session);
                disp(session_dir);

                spm_jobman('initcfg');

                clear jobs
                matlabbatch={};

                spm_dir=fullfile(session_dir, 'spm');
                run_files={};
                spm_files=dir(spm_dir);
                for f=3:length(spm_files)
                    if strcmp(spm_files(f).name(1:3),'spm') && length(strfind(spm_files(f).name,'-motor-epo.mat'))
                        run_files{end+1}=fullfile(spm_dir, spm_files(f).name);
                        disp(spm_files(f).name);
                    end
                end
                m_idx=1;
                merged_file=fullfile(spm_dir, sprintf('cspm_converted_autoreject-%s-%s-motor-epo.mat',subject,session));
                if length(run_files)>1
                    matlabbatch{m_idx}.spm.meeg.preproc.merge.D = run_files';
                    matlabbatch{m_idx}.spm.meeg.preproc.merge.recode.file = '.*';
                    matlabbatch{m_idx}.spm.meeg.preproc.merge.recode.labelorg = '.*';
                    matlabbatch{m_idx}.spm.meeg.preproc.merge.recode.labelnew = '#labelorg#';
                    matlabbatch{m_idx}.spm.meeg.preproc.merge.prefix = 'c';
                    m_idx=m_idx+1;

                    matlabbatch{m_idx}.spm.meeg.other.copy.D(1) = cfg_dep('Merging: Merged Datafile', substruct('.','val', '{}',{m_idx-1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
                    matlabbatch{m_idx}.spm.meeg.other.copy.outfile = merged_file;    
                    m_idx=m_idx+1;
                else
                    matlabbatch{m_idx}.spm.meeg.other.copy.D = run_files(1);
                    matlabbatch{m_idx}.spm.meeg.other.copy.outfile = merged_file;
                    m_idx=m_idx+1;
                end

                matlabbatch{m_idx}.spm.meeg.preproc.crop.D(1) = cfg_dep('Copy: Copied M/EEG datafile', substruct('.','val', '{}',{m_idx-1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
                matlabbatch{m_idx}.spm.meeg.preproc.crop.timewin = [-1000 1000];
                matlabbatch{m_idx}.spm.meeg.preproc.crop.freqwin = [-Inf Inf];
                matlabbatch{m_idx}.spm.meeg.preproc.crop.channels{1}.all = 'all';
                matlabbatch{m_idx}.spm.meeg.preproc.crop.prefix = 'p';
                m_idx=m_idx+1;

                matlabbatch{m_idx}.spm.meeg.averaging.average.D(1) = cfg_dep('Copy: Copied M/EEG datafile', substruct('.','val', '{}',{m_idx-2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
                matlabbatch{m_idx}.spm.meeg.averaging.average.userobust.standard = false;
                matlabbatch{m_idx}.spm.meeg.averaging.average.plv = false;
                matlabbatch{m_idx}.spm.meeg.averaging.average.prefix = 'm';
                m_idx=m_idx+1;

                matlabbatch{m_idx}.spm.meeg.preproc.crop.D(1) = cfg_dep('Averaging: Averaged Datafile', substruct('.','val', '{}',{m_idx-1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
                matlabbatch{m_idx}.spm.meeg.preproc.crop.timewin = [-1000 1000];
                matlabbatch{m_idx}.spm.meeg.preproc.crop.freqwin = [-Inf Inf];
                matlabbatch{m_idx}.spm.meeg.preproc.crop.channels{1}.all = 'all';
                matlabbatch{m_idx}.spm.meeg.preproc.crop.prefix = 'p';


                spm_jobman('run', matlabbatch);

                spm_jobman('initcfg');

                clear jobs
                matlabbatch={};

                spm_dir=fullfile(session_dir, 'spm');
                run_files={};
                spm_files=dir(spm_dir);
                for f=3:length(spm_files)
                    if strcmp(spm_files(f).name(1:3),'spm') && length(strfind(spm_files(f).name,'-visual-epo.mat'))
                        run_files{end+1}=fullfile(spm_dir, spm_files(f).name);
                        disp(spm_files(f).name);
                    end
                end
                merged_file=fullfile(spm_dir, sprintf('cspm_converted_autoreject-%s-%s-visual-epo.mat',subject,session));
                m_idx=1;
                if length(run_files)>1
                    matlabbatch{m_idx}.spm.meeg.preproc.merge.D = run_files';
                    matlabbatch{m_idx}.spm.meeg.preproc.merge.recode.file = '.*';
                    matlabbatch{m_idx}.spm.meeg.preproc.merge.recode.labelorg = '.*';
                    matlabbatch{m_idx}.spm.meeg.preproc.merge.recode.labelnew = '#labelorg#';
                    matlabbatch{m_idx}.spm.meeg.preproc.merge.prefix = 'c';
                    m_idx=m_idx+1;

                    matlabbatch{m_idx}.spm.meeg.other.copy.D(1) = cfg_dep('Merging: Merged Datafile', substruct('.','val', '{}',{m_idx-1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
                    matlabbatch{m_idx}.spm.meeg.other.copy.outfile = merged_file;
                    m_idx=m_idx+1;
                else
                    matlabbatch{m_idx}.spm.meeg.other.copy.D = run_files(1);
                    matlabbatch{m_idx}.spm.meeg.other.copy.outfile = merged_file;
                    m_idx=m_idx+1;
                end

                matlabbatch{m_idx}.spm.meeg.averaging.average.D(1) = cfg_dep('Copy: Copied M/EEG datafile', substruct('.','val', '{}',{m_idx-1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
                matlabbatch{m_idx}.spm.meeg.averaging.average.userobust.standard = false;
                matlabbatch{m_idx}.spm.meeg.averaging.average.plv = false;
                matlabbatch{m_idx}.spm.meeg.averaging.average.prefix = 'm';
                m_idx=m_idx+1;

                matlabbatch{m_idx}.spm.meeg.preproc.crop.D(1) = cfg_dep('Averaging: Averaged Datafile', substruct('.','val', '{}',{m_idx-1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
                matlabbatch{m_idx}.spm.meeg.preproc.crop.timewin = [-500 1000];
                matlabbatch{m_idx}.spm.meeg.preproc.crop.freqwin = [-Inf Inf];
                matlabbatch{m_idx}.spm.meeg.preproc.crop.channels{1}.all = 'all';
                matlabbatch{m_idx}.spm.meeg.preproc.crop.prefix = 'rdk_p';
                m_idx=m_idx+1;

                matlabbatch{m_idx}.spm.meeg.preproc.crop.D(1) = cfg_dep('Averaging: Averaged Datafile', substruct('.','val', '{}',{m_idx-2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
                matlabbatch{m_idx}.spm.meeg.preproc.crop.timewin = [1500 3000];
                matlabbatch{m_idx}.spm.meeg.preproc.crop.freqwin = [-Inf Inf];
                matlabbatch{m_idx}.spm.meeg.preproc.crop.channels{1}.all = 'all';
                matlabbatch{m_idx}.spm.meeg.preproc.crop.prefix = 'instr_p';

                spm_jobman('run', matlabbatch);
            end
        end
    end
end