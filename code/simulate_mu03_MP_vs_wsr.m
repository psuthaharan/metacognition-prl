function simulate_mu03_MP_vs_wsr()
% simulate_mu03_MP_vs_wsr
% ---------------------------------------------------------------
%
% This implements:
%   - For each subject, keep all fitted HGF + softmax parameters
%     fixed except the prior volatility mu03.
%   - Systematically vary mu03, simulate full PRL sessions with
%     paella_sim_alltrials, compute win-switch rate (WSR).
%   - Average the mu03 -> WSR curves across subjects, and then
%     split subjects by metacognitive score (MP) to show
%     modulation of this mapping.
%
% Requirements (on MATLAB path):
%   - paella_sim_alltrials.m
%   - raw-data/HGF_fits_allSubjects_V3.mat   (variable: out)
%   - raw-data/mp_scores_continuous_500.csv
%
% HGF_fits_allSubjects_V3.mat:
%   out(1,:)       : header row (includes 'Subject'/'Subjects',
%                    'est', 'mu03', etc.)
%   out(2:end,1)   : subject ID (mturk_id or similar)
%   out(2:end,est) : TAPAS fit struct (used by paella_sim_alltrials)
%   out(2:end,mu03): fitted mu03 values
%
% mp_scores_continuous_500.csv:
%   - must contain columns:
%       mturk_id
%       comprehension_continuous
%       judgment_continuous
%       evaluation_continuous
%       final_decision_continuous
%       confidence_continuous
%   - we compute MP_avg as the mean of those five columns.
%
% ---------------------------------------------------------------

%% ------------------------------------------------------------------------
% 0. File locations and basic settings
% -------------------------------------------------------------------------
mat_file = fullfile('raw-data', 'HGF_fits_allSubjects_V3.mat');
csv_file = fullfile('raw-data', 'mp_scores_continuous_500.csv');
species  = 'human';      % paella_sim_alltrials species flag

% For reproducible, smooth curves
rng(1);          % fixed random seed
n_iter   = 100;           % sims per (subject, mu03 point)
n_mu     = 15;           % number of mu03 grid points

fprintf('>>> Loading HGF fits from %s\n', mat_file);
S = load(mat_file);
if ~isfield(S, 'out')
    error('Expected variable ''out'' in %s.', mat_file);
end
out = S.out;

[n_rows, n_cols] = size(out);
fprintf('Loaded ''out'' with %d rows and %d columns.\n', n_rows, n_cols);

%% ------------------------------------------------------------------------
% 1. Identify relevant columns in "out"
% -------------------------------------------------------------------------
header = out(1, :);  % first row is header

hdr2str = @(c) char(c{1});  % helper to extract header cell as char

subjectCol = [];
paramCol   = [];
mu03Col    = [];

for j = 1:n_cols
    name_j = strtrim(hdr2str(header(j)));  % remove stray spaces

    % subject id column: allow several possible labels
    if any(strcmpi(name_j, {'Subject', 'Subjects', 'mturk_id'}))
        subjectCol = j;
    elseif strcmpi(name_j, 'est')
        paramCol = j;
    elseif strcmpi(name_j, 'mu03')
        mu03Col = j;
    end
end

if isempty(subjectCol) || isempty(paramCol) || isempty(mu03Col)
    error(['Could not detect Subject(/s)/mturk_id or est or mu03 ' ...
           'columns in ''out'' header.']);
end

fprintf('Detected columns: SubjectCol=%d, est=%d, mu03=%d\n', ...
    subjectCol, paramCol, mu03Col);

% Data rows (skip header)
dataRows = 2:n_rows;

% Extract ids, parameter structs, and fitted mu03
mturk_id_out = cell(numel(dataRows), 1);
param_out    = cell(numel(dataRows), 1);
mu03_fit_out = nan(numel(dataRows), 1);
valid_idx    = false(numel(dataRows), 1);

for ii = 1:numel(dataRows)
    r = dataRows(ii);

    mturk_id_out{ii} = char(out{r, subjectCol});
    this_param       = out{r, paramCol};
    this_mu03        = out{r, mu03Col};

    if isstruct(this_param) && ~isempty(this_mu03) && ~isnan(this_mu03)
        param_out{ii}    = this_param;
        mu03_fit_out(ii) = this_mu03;
        valid_idx(ii)    = true;
    end
end

% Keep only rows with a valid HGF fit struct and mu03 value
mturk_id_out = mturk_id_out(valid_idx);
param_out    = param_out(valid_idx);
mu03_fit_out = mu03_fit_out(valid_idx);

n_sub_raw = numel(mturk_id_out);
fprintf('Using %d participants with valid HGF fits before MP merge.\n', n_sub_raw);

%% ------------------------------------------------------------------------
% 2. Load MP / paranoia data, compute MP_avg, and align subjects
% -------------------------------------------------------------------------
fprintf('>>> Loading MP / paranoia data from %s\n', csv_file);
T = readtable(csv_file);

% Compute MP_avg as the mean of the five continuous MP dimensions
mp_dims = {'comprehension_continuous', 'judgment_continuous', ...
           'evaluation_continuous', 'final_decision_continuous', ...
           'confidence_continuous'};

missing_cols = setdiff(mp_dims, T.Properties.VariableNames);
if ~isempty(missing_cols)
    error('Missing expected MP columns in CSV: %s', strjoin(missing_cols, ', '));
end

T.MP_avg = mean(T{:, mp_dims}, 2, 'omitnan');

% Align by mturk_id 
csv_ids = cellstr(string(T.mturk_id));
out_ids = mturk_id_out;

[common_ids, idx_out, idx_csv] = intersect(out_ids, csv_ids, 'stable'); %#ok

fprintf('Matched %d participants across HGF fits and MP dataset.\n', numel(idx_out));

param_out    = param_out(idx_out);
mu03_fit_out = mu03_fit_out(idx_out);
MP_avg       = T.MP_avg(idx_csv);

% Report mu03 range
fprintf('Observed fitted mu03 range: [%.3f, %.3f], median = %.3f\n', ...
    min(mu03_fit_out), max(mu03_fit_out), median(mu03_fit_out));

n_sub = numel(param_out); % update after matching

%% ------------------------------------------------------------------------
% 3. Define mu03 grid (avoid unstable extremes)
% -------------------------------------------------------------------------
lo = prctile(mu03_fit_out, 15);   % 15th percentile (slightly conservative)
hi = prctile(mu03_fit_out, 85);   % 85th percentile
mu03_grid = linspace(lo, hi, n_mu);

fprintf('mu03 grid from %.3f to %.3f (%d points).\n', lo, hi, n_mu);

%% ------------------------------------------------------------------------
% 4. Main simulation loop - for each subject & mu03 value
% -------------------------------------------------------------------------
wsr_all      = nan(n_sub, n_mu);   % mean simulated WSR per subject & mu03
fail_counter = 0;                  % how many subject x mu03 combos failed

h = waitbar(0, 'Simulating HGF across mu03 values...');
total_steps = n_sub * n_mu;
step_count  = 0;

for s = 1:n_sub
    base_param = param_out{s};

    for im = 1:n_mu
        mu03_val = mu03_grid(im);

        % Copy parameter struct and overwrite mu_0(3) with new mu03
        param = base_param;
        if numel(param.p_prc.mu_0) < 3
            % This should not happen, but guard just in case
            fail_counter = fail_counter + 1;
            continue;
        end
        param.p_prc.mu_0(3) = mu03_val;

        % Repeat simulations to average over stochasticity
        wsr_rep = nan(n_iter, 1);

        for it = 1:n_iter
            try
                [y_sim, u_sim] = paella_sim_alltrials(param, species);

                % Drop dummy trial (1st row)
                y = y_sim(2:end);
                u = u_sim(2:end);

                % Compute win-switch rate
                wsr_rep(it) = compute_wsr(y, u);

            catch
                % If paella_sim_alltrials throws (e.g. negative precision),
                % mark this repetition as NaN and move on.
                fail_counter = fail_counter + 1;
                wsr_rep(it)  = NaN;
                % We do NOT rethrow or warn, to keep output clean.
            end
        end

        wsr_all(s, im) = mean(wsr_rep, 'omitnan');

        % Update progress bar
        step_count = step_count + 1;
        if mod(step_count, 20) == 0 || step_count == total_steps
            waitbar(step_count / total_steps, h, ...
                sprintf('Simulating... %d / %d combos', step_count, total_steps));
        end
    end
end
close(h);

fprintf('Total simulation failures (subject x mu03 x repetition): %d\n', fail_counter);

%% ------------------------------------------------------------------------
% 5. Aggregate across subjects and split by MP (z-scored, ±1 SD)
% -------------------------------------------------------------------------
% Overall mean mapping (all participants)
mean_wsr_all = mean(wsr_all, 1, 'omitnan');

% z-score MP 
MP_z = (MP_avg - mean(MP_avg, 'omitnan')) ./ std(MP_avg, [], 'omitnan');

low_thr  = -1;   % -1 SD
high_thr =  1;   % +1 SD

is_low_MP  = MP_z <= low_thr;
is_high_MP = MP_z >= high_thr;

fprintf('Low MP group (<= -1 SD):  n = %d\n', sum(is_low_MP));
fprintf('High MP group (>= +1 SD): n = %d\n', sum(is_high_MP));

mean_wsr_low  = mean(wsr_all(is_low_MP,  :), 1, 'omitnan');
mean_wsr_high = mean(wsr_all(is_high_MP, :), 1, 'omitnan');

%% ------------------------------------------------------------------------
% 5b. Quantify and test slope difference (dampening effect)
% -------------------------------------------------------------------------
x = mu03_grid(:);

% Fit simple linear slopes to group means
p_low  = polyfit(x, mean_wsr_low(:),  1);
p_high = polyfit(x, mean_wsr_high(:), 1);

slope_low  = p_low(1);
slope_high = p_high(1);

% Build trial-level table for interaction test
Y = [mean_wsr_low(:); mean_wsr_high(:)];
X = [x; x];
G = [zeros(numel(x),1); ones(numel(x),1)]; % 0 = low MP, 1 = high MP

tbl = table(Y, X, G, 'VariableNames', {'WSR','mu03','MPgroup'});
mdl = fitlm(tbl, 'WSR ~ mu03*MPgroup');

p_interaction = mdl.Coefficients.pValue(strcmp(mdl.Coefficients.Properties.RowNames, 'mu03:MPgroup'));

fprintf('Slope Low MP  = %.4f\n', slope_low);
fprintf('Slope High MP = %.4f\n', slope_high);
fprintf('Slope difference interaction p = %.4g\n', p_interaction);


%% ------------------------------------------------------------------------
% 6. Plot
% -------------------------------------------------------------------------
figure('Color', 'w', 'Position', [100 100 1200 480]);

% Panel A: overall mapping
subplot(1,2,1);
plot(mu03_grid, mean_wsr_all, '-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('\mu_{03} (prior volatility)');
ylabel('Model-predicted win-switch rate (WSR)');
title('HGF-predicted mapping: \mu_{03} \rightarrow WSR (all participants)');
grid on;
xlim([min(mu03_grid) max(mu03_grid)]);

% Panel B: MP modulation
subplot(1,2,2);
plot(mu03_grid, mean_wsr_low,  '-o', 'LineWidth', 2, 'MarkerSize', 6); hold on;
plot(mu03_grid, mean_wsr_high, '-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('\mu_{03} (prior volatility)');
ylabel('Model-predicted win-switch rate (WSR)');
title('MP modulation of \mu_{03} \rightarrow WSR mapping');
legend({'Low MP', 'High MP'}, 'Location', 'best');
grid on;
xlim([min(mu03_grid) max(mu03_grid)]);

sgtitle('HGF simulations: volatility-belief to behaviour mapping');

fprintf('Simulation complete. Plots generated.\n');

end

% ========================================================================
% Function to compute win-switch rate (WSR)
% ========================================================================
function wsr = compute_wsr(y, u)
% y : choices (1..3)
% u : outcomes (0/1)
% WSR: proportion of rewarded trials followed by a switch.

if numel(y) ~= numel(u)
    error('y and u must be the same length.');
end

n = numel(y);
if n < 2
    wsr = NaN;
    return;
end

win_trials = find(u(1:end-1) == 1);   % indices t where trial t was rewarded
if isempty(win_trials)
    wsr = NaN;
    return;
end

switched = y(win_trials + 1) ~= y(win_trials);
wsr      = mean(switched);

end
