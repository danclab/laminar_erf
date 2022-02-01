dataset_path='/media/bonaiuto/TOSHIBA EXT/cued_action_meg/';
subj_id='sub-001';
session_id='ses-01';
epo='motor_right';

data_dir=fullfile(dataset_path,'derivatives/processed',subj_id, session_id);
output_dir=fullfile(data_dir, 'spm');
run_source_tc=[];
run_f_diff=[];
for i=1:3
    load(fullfile(output_dir, sprintf('invert_mspm_converted_autoreject-%s-%s-%03d-%s-epo_localizer_results.mat',subj_id,session_id,i,epo)));
    run_source_tc(i,:)=invert_localizer_results.clusters.source_tc;
    s_times=invert_localizer_results.times.*1000;
    
    load(fullfile(output_dir, sprintf('invert_mspm_converted_autoreject-%s-%s-%03d-%s-epo_tc_results.mat',subj_id,session_id,i,epo)));
    run_f_diff(i,:)=invert_tc_results.clusters.f_diff;
    f_times=invert_tc_results.times(invert_tc_results.left_idx:invert_tc_results.right_idx);    
end

figure();
subplot(2,1,1);
hold all;
for i=1:3
    plot(s_times,run_source_tc(i,:));
end
xlim([s_times(1) s_times(end)]);
xlabel('Time (ms)')
ylabel('Source density');
subplot(2,1,2);
hold all;
for i=1:3
    plot(f_times,run_f_diff(i,:));
end
plot([f_times(1) f_times(end)],[0 0],'k');
plot([f_times(1) f_times(end)],[3 3],'k--');
plot([f_times(1) f_times(end)],[-3 -3],'k--');
xlim([s_times(1) s_times(end)]);
xlabel('Time (ms)')
ylabel('Fpial-Fwhite');
