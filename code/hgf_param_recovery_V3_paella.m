% Parameter recovery for the V3 PRL HGF fits.
function recovery = hgf_param_recovery_V3_paella(nSubsUse, nSim)

% hgf_param_recovery_V3_paella
% -----------------------------

%
% Uses:
%   - raw-data/HGF_fits_allSubjects_V3.mat   
%   - paella_sim_alltrials.m                (custom simulator)
%
% For each subject:
%   1. Take the fitted HGF parameters ("TRUE" parameters).
%   2. Simulate full PRL performance (choices + outcomes) from those params.
%   3. Refit the SAME HGF+softmax model to the simulated data.
%   4. Repeat steps 2–3 for nSim iterations.
%   5. Store the recovered parameters for each simulation and their mean.
%
% At the end:
%   - Saves a .mat file with TRUE and RECOVERED parameters.
%   - Returns a 'recovery' struct with all information.
%   - Reports how many subjects had at least one successful recovery,
%     and how many encountered NegPostPrec / VarApprox errors.
%   - Plots:
%       (Left)  Fitted vs mean recovered mu03 with correlation.
%       (Right) 5x5 correlation matrix of TRUE vs RECOVERED parameters.
%
% USAGE EXAMPLES:
%   % Quick test: 5 subjects, 10 simulations each
%   recovery = hgf_param_recovery_V3_paella(5, 10);
%
%   % Full run: all subjects, 50 simulations each
%   recovery = hgf_param_recovery_V3_paella([], 50);

% -------------------------------------------------------------------------
% 0. Arguments and paths
% -------------------------------------------------------------------------
if nargin < 2 || isempty(nSim)
    nSim = 20;          % default simulations/subject
end

% Load fitted HGF results
baseDir    = pwd;
rawDataDir = fullfile(baseDir, 'raw-data');
fitPath    = fullfile(rawDataDir, 'HGF_fits_allSubjects_V3.mat');

if ~isfile(fitPath)
    error('Could not find HGF fits file:\n  %s', fitPath);
end

S = load(fitPath);   % should contain variable 'out'
if ~isfield(S, 'out')
    error('File %s does not contain variable ''out''.', fitPath);
end
out = S.out;

% Number of subjects in fitted file
nSubsAll = size(out, 1) - 1;   % minus header row

if nargin < 1 || isempty(nSubsUse) || nSubsUse > nSubsAll
    nSubsUse = nSubsAll;
end

fprintf('Parameter recovery: using %d of %d subjects, %d simulations each.\n', ...
        nSubsUse, nSubsAll, nSim);

% -------------------------------------------------------------------------
% 1. Parameter containers
% -------------------------------------------------------------------------
paramNames = {'mu02','mu03','kappa2','omega2','omega3'};
nParam     = numel(paramNames);

subjectIDs     = cell(nSubsUse, 1);
trueParams     = NaN(nSubsUse, nParam);          % TRUE (fitted) parameters
recParamsAll   = NaN(nSubsUse, nParam, nSim);    % RECOVERED per sim
recParamsMean  = NaN(nSubsUse, nParam);          % mean over sims

% Tracking success / errors
simSuccess         = false(nSubsUse, nSim);   % did this sim ever succeed?
subjectHadNegError = false(nSubsUse, 1);      % any NegPostPrec / VarApprox per subject

% -------------------------------------------------------------------------
% 2. Progress bar setup
% -------------------------------------------------------------------------
totalSteps = nSubsUse * nSim;
step       = 0;
tStart     = tic;

h = waitbar(0, 'Initializing parameter recovery...');

% --- Resize waitbar so that it widens LEFT --------------------------------
wbpos   = get(h, 'Position');   % [left bottom width height]
expandBy = 10;                 % how much wider you want it

wbpos(1) = wbpos(1) - expandBy; % shift window left
wbpos(3) = wbpos(3) + expandBy; % widen window
wbpos(4) = 70;                 % a bit taller for multi-line text

set(h, 'Position', wbpos);
drawnow;

% How many times we are willing to re-simulate+refit if TAPAS throws
% "Negative posterior precision" or similar errors for a given sim.
maxAttemptsPerSim = 1000;   % you can adjust if you like

% -------------------------------------------------------------------------
% 3. Main recovery loop
% -------------------------------------------------------------------------
for s = 1:nSubsUse

    subjID         = out{s+1, 1};   % subject label string
    est_true       = out{s+1, 3};   % fitted HGF struct for this subject
    subjectIDs{s}  = subjID;

    % ---- Extract TRUE (fitted) parameters --------------------------------
    true_mu02   = est_true.p_prc.mu_0(2);
    true_mu03   = est_true.p_prc.mu_0(3);
    true_kappa2 = est_true.p_prc.ka(2);
    true_omega2 = est_true.p_prc.om(2);
    true_omega3 = est_true.p_prc.om(3);

    trueParams(s,:) = [true_mu02 true_mu03 true_kappa2 true_omega2 true_omega3];

    for k = 1:nSim
        step = step + 1;

        % ---- Progress bar message ---------------------------------------
        frac    = step / totalSteps;
        elapsed = toc(tStart);
        if step == 1
            tRemain = NaN;
        else
            tRemain = elapsed * (totalSteps - step) / step;
        end

        % ---- How many subjects are "recovered" vs "skipped" so far? ----
        % We only count subjects fully completed *before* the current one.
        if s > 1
            % Any success across all sims for each completed subject
            completedSuccess = any(simSuccess(1:s-1, :), 2);
            recoveredSoFar   = sum(completedSuccess);
            skippedSoFar     = (s-1) - recoveredSoFar;
        else
            recoveredSoFar = 0;
            skippedSoFar   = 0;
        end

        % Multi-line waitbar text: line 1 = Subj/Sim, line 2 = counts,
        % line 3 = timing info.
        msg = sprintf([ ...
            'Subj %d/%d | Sim %d/%d\n' ...
            'Recovered subj so far: %d | Skipped subj so far: %d\n' ...
            'Elapsed %s | Remaining %s'], ...
            s, nSubsUse, k, nSim, ...
            recoveredSoFar, skippedSoFar, ...
            formatTime(elapsed), formatTime(tRemain));

        waitbar(frac, h, msg);

        % ---- Simulate and refit with RETRIES ---------------------------
        success    = false;
        lastErrMsg = '';
        lastErrID  = '';

        % >>> RETRY LOOP START
        for attempt = 1:maxAttemptsPerSim
            try
                % 1) Simulate full-trial choices & outcomes from TRUE parameters
                [y_sim, u_sim] = paella_sim_alltrials(est_true, 'human');

                % Drop dummy "zeroth" trial
                y_sim = y_sim(2:end);
                u_sim = u_sim(2:end);

                % 2) Refit SAME model to the simulated data
                est_rec = tapas_fitModel(y_sim, u_sim, ...
                                         'tapas_hgf_ar1_binary_mab_config_1', ...
                                         'tapas_softmax_mu3_config');

                % If we get here, both simulation and fit succeeded
                success = true;
                break;

            catch ME
                lastErrMsg = ME.message;
                lastErrID  = ME.identifier;

                % Flag this subject as having encountered a precision / VAR error
                if contains(lastErrID, 'NegPostPrec')          || ...
                   contains(lastErrID, 'VarApproxInvalid')     || ...
                   contains(lastErrMsg, 'NegPostPrec')         || ...
                   contains(lastErrMsg, 'Variational approximation invalid')

                    subjectHadNegError(s) = true;
                end

                fprintf('  Attempt %d failed for sub %s (index %d), sim %d: %s\n', ...
                        attempt, subjID, s, k, ME.message);
                % loop continues to another simulate+fit attempt
            end
        end
        % <<< RETRY LOOP END

        if ~success
            warning(['Recovery failed for subject %s (sub %d), sim %d after %d attempts.\n' ...
                     '  Last error: %s'], ...
                     subjID, s, k, maxAttemptsPerSim, lastErrMsg);
            recParamsAll(s,:,k) = NaN(1, nParam);
            % simSuccess remains false for this (s,k)
            continue;
        end

        % Mark this simulation as successful
        simSuccess(s,k) = true;

        % Recovered parameters for this (successful) sim
        rec_mu02   = est_rec.p_prc.mu_0(2);
        rec_mu03   = est_rec.p_prc.mu_0(3);
        rec_kappa2 = est_rec.p_prc.ka(2);
        rec_omega2 = est_rec.p_prc.om(2);
        rec_omega3 = est_rec.p_prc.om(3);

        recParamsAll(s,:,k) = [rec_mu02 rec_mu03 rec_kappa2 rec_omega2 rec_omega3];
    end

    % Mean of recovered parameters across simulations for this subject
    recParamsMean(s,:) = mean(recParamsAll(s,:,:), 3, 'omitnan');
end

close(h);

% -------------------------------------------------------------------------
% 4. Compute subject-level recovery stats
% -------------------------------------------------------------------------
subjectAnySuccess   = any(simSuccess, 2);       % at least one successful sim
nSubjectsRecovered  = sum(subjectAnySuccess);
nSubjectsNoRecovery = nSubsUse - nSubjectsRecovered;

nSubjectsWithErrors = sum(subjectHadNegError);

fprintf('\nRecovery summary:\n');
fprintf('  Subjects with ≥1 successful recovery: %d / %d\n', ...
        nSubjectsRecovered, nSubsUse);
fprintf('  Subjects with 0 successful recoveries: %d / %d\n', ...
        nSubjectsNoRecovery, nSubsUse);
fprintf('  Subjects that encountered NegPostPrec / VarApprox errors in recovery: %d / %d\n', ...
        nSubjectsWithErrors, nSubsUse);

% -------------------------------------------------------------------------
% 5. Compute correlations
% -------------------------------------------------------------------------
% TRUE vs mean RECOVERED per parameter
R = corr(trueParams, recParamsMean, 'rows', 'pairwise');  % 5x5 matrix
[rMu03, ~] = corr(trueParams(:,2), recParamsMean(:,2), 'rows', 'complete');

fprintf('\nmu03 recovery correlation (TRUE vs mean recovered): r = %.3f\n', rMu03);

% -------------------------------------------------------------------------
% 6. Save everything to .mat
% -------------------------------------------------------------------------
savePath = fullfile(rawDataDir, ...
    sprintf('HGF_paramRecovery_V3_paella_%dSubs_%dSim.mat', nSubsUse, nSim));

recovery = struct();
recovery.subjectIDs          = subjectIDs;
recovery.paramNames          = paramNames;
recovery.trueParams          = trueParams;
recovery.recParamsAll        = recParamsAll;
recovery.recParamsMean       = recParamsMean;
recovery.R                   = R;
recovery.rMu03               = rMu03;
recovery.nSubsUse            = nSubsUse;
recovery.nSim                = nSim;
recovery.maxAttemptsPerSim   = maxAttemptsPerSim;

% Diagnostics:
recovery.simSuccess          = simSuccess;         % nSubsUse x nSim logical
recovery.subjectAnySuccess   = subjectAnySuccess;  % nSubsUse x 1 logical
recovery.nSubjectsRecovered  = nSubjectsRecovered;
recovery.nSubjectsNoRecovery = nSubjectsNoRecovery;
recovery.subjectHadNegError  = subjectHadNegError; % nSubsUse x 1 logical
recovery.nSubjectsWithErrors = nSubjectsWithErrors;

save(savePath, 'recovery');
fprintf('Saved recovery results to:\n  %s\n', savePath);

% -------------------------------------------------------------------------
% 7. Plot: mu03 recovery + 5x5 correlation matrix
% -------------------------------------------------------------------------
figure('Color','w','Position',[100 100 1200 600]);

% --- Panel 1: scatter of TRUE vs mean RECOVERED mu03 --------------------
subplot(1,2,1);

true_mu03 = trueParams(:,2);
rec_mu03  = recParamsMean(:,2);

scatter(true_mu03, rec_mu03, 40, 'filled');
hold on;

% unity line
lims = [min([true_mu03; rec_mu03]) max([true_mu03; rec_mu03])];
if diff(lims) == 0
    lims = lims + [-1 1]; % avoid zero-width axes
end
plot(lims, lims, 'k--', 'LineWidth', 1);
xlim(lims); ylim(lims);

xlabel('Fitted \mu_{03}','FontSize',12);
ylabel('Recovered \mu_{03} (mean across sims)','FontSize',12);
title(sprintf('\\mu_{03} recovery (r = %.3f)', rMu03), 'FontSize',14, 'FontWeight','bold');
box on; grid on;

% --- Panel 2: TRUE vs RECOVERED correlation matrix ----------------------
subplot(1,2,2);

imagesc(R, [-1 1]);
axis square;
colorbar;

xticks(1:nParam);
yticks(1:nParam);

xticklabels(strcat('Rec', paramNames));
yticklabels(strcat('True', paramNames));

xtickangle(45);

title('TRUE vs MEAN RECOVERED parameter correlations', ...
      'FontSize',14, 'FontWeight','bold');

set(gca, 'FontSize', 11);

end % main function

% -------------------------------------------------------------------------
% Helper: format seconds as mm:ss or hh:mm:ss
% -------------------------------------------------------------------------
function sOut = formatTime(t)
    if isnan(t)
        sOut = '--:--';
        return;
    end
    if t < 3600
        mm = floor(t/60);
        ss = round(rem(t,60));
        sOut = sprintf('%02d:%02d', mm, ss);
    else
        hh = floor(t/3600);
        mm = floor(rem(t,3600)/60);
        ss = round(rem(t,60));
        sOut = sprintf('%02d:%02d:%02d', hh, mm, ss);
    end
end
