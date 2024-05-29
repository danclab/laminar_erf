function D = subset_trials(D, fname, trial_idx)

Dnew = clone(D, fname, [D.nchannels D.nsamples length(trial_idx)]);

for i = 1:length(trial_idx)
    
    Dnew(:, :, i) =  D(:, :, trial_idx(i));
end  %

%-Copy trial-specific data.
%--------------------------------------------------------------------------
Dnew = conditions(Dnew, ':', conditions(D, trial_idx));
Dnew = repl(Dnew, ':', repl(D, trial_idx));
Dnew = events(Dnew, ':', events(D, trial_idx));
Dnew = trialonset(Dnew, ':', trialonset(D, trial_idx));

save(Dnew);

D = Dnew;
