% CUSTOMIZED SIMULATION SCRIPT FOR HGF-PRL (also used in Suthaharan et al.,
% 2024)
% simulate choices and outcomes from HGF belief parameters
function [y,u] = paella_sim_alltrials(param, species)
% paella_sim_alltrials
% ---------------------
% Simulate FULL-session PRL performance (all trials) from an HGF+softmax
% fit ("param"). This preserves:
%   - multi-reversal schedule
%   - performance-dependent reversal (9/10 best-choice rule)
%   - subject-specific HGF / softmax parameters
%
% INPUTS
%   param   : struct returned by tapas_fitModel for this subject
%   species : 'monkey' (uses stabl.txt) or anything else for humans
%
% OUTPUTS
%   y       : simulated choices  (length = n_trials + 1, incl. dummy)
%   u       : simulated outcomes (length = n_trials + 1, incl. dummy)
%
%   NOTE: use y(2:end), u(2:end) as the actual simulated data.

% -------------------------------------------------------------------------
% Initialization
n_trials  = size(param.y(~isnan(param.y)), 1);   % number of real trials
y         = zeros(n_trials, 1);
u         = zeros(n_trials, 1);

n_levels  = param.c_prc.n_levels;
n_bandits = param.c_prc.n_bandits;

prc_model = param.c_prc.model; %#ok
obs_model = param.c_obs.model; %#ok

param.ign = [];
param.irr = [];

% Coupled updating
coupled = false;
if param.c_prc.coupled
    if n_bandits == 2
        coupled = true;
    else
        error('tapas:hgf:HgfBinaryMab:CoupledOnlyForTwo', ...
              'Coupled updating can only be configured for 2 bandits.');
    end
end

% -------------------------------------------------------------------------
% Set up estimated parameters
mu_0 = param.p_prc.mu_0;
sa_0 = param.p_prc.sa_0; %#ok
phi  = param.p_prc.phi;
m    = param.p_prc.m;
ka   = param.p_prc.ka;
om   = param.p_prc.om;
th   = exp(param.p_prc.om(3));

% Time axis
if param.c_prc.irregular_intervals
    if size(u,2) > 1
        t = [0; param.u(:,end)];
    else
        error('tapas:hgf:InputSingleColumn', ...
              'Input matrix must contain more than one column if irregular_intervals is true.');
    end
else
    t = ones(n_trials+1, 1);
end

% -------------------------------------------------------------------------
% Initialization of learning variables and priors
mu   = NaN(n_trials+1, n_levels,   n_bandits);
pi   = NaN(n_trials+1, n_levels,   n_bandits);
muhat= NaN(n_trials+1, n_levels,   n_bandits);
pihat= NaN(n_trials+1, n_levels,   n_bandits);
v    = NaN(n_trials+1, n_levels);
w    = NaN(n_trials+1, n_levels-1);
da   = NaN(n_trials+1, n_levels);

% Priors (trial 0)
mu(1,1,:)      = tapas_sgm(param.p_prc.mu_0(2), 1);
muhat(1,1,:)   = mu(1,1,:);
pihat(1,1,:)   = 0;
pi(1,1,:)      = Inf;
mu(1,2:end,:)  = repmat(param.p_prc.mu_0(2:end), [1 1 n_bandits]);
pi(1,2:end,:)  = repmat(1./param.p_prc.mu_0(2:end), [1 1 n_bandits]);

% Dummy zeroth trial
u = [0; u];
y = [1; y];

infStates = NaN(n_trials+1, n_levels, n_bandits, 4);
traj_sim  = struct();

% -------------------------------------------------------------------------
% Set up multi-reversal PRL task agent-environment
prob_set1 = [0.9 0.5 0.1; 0.9 0.1 0.5; 0.5 0.9 0.1; 0.5 0.1 0.9; 0.1 0.9 0.5; 0.1 0.5 0.9];
prob_set2 = [0.8 0.4 0.2; 0.8 0.2 0.4; 0.4 0.8 0.2; 0.4 0.2 0.8; 0.2 0.8 0.4; 0.2 0.4 0.8];

change_contingency = false;
previous_prob_row1 = 0;   % for trials 1–80
previous_prob_row2 = 0;   % for trials 81–160

% Monkey reward schedule (from Suthaharan et al., 2024)
if strcmpi(species, 'monkey')
    prob_reinforced = readtable('stabl.txt');
    prob_reinforced = prob_reinforced(1:n_trials, :);
    prob_reinforced = [{0 0 0}; prob_reinforced];  % dummy row
end

% -------------------------------------------------------------------------
% Simulate loop over trials
for i_trials = 2:(n_trials+1)
    if ismember(i_trials-1, param.ign)
        % Ignored trial: carry forward everything
        mu(i_trials,:,:)    = mu(i_trials-1,:,:);
        pi(i_trials,:,:)    = pi(i_trials-1,:,:);
        muhat(i_trials,:,:) = muhat(i_trials-1,:,:);
        pihat(i_trials,:,:) = pihat(i_trials-1,:,:);
        v(i_trials,:)       = v(i_trials-1,:);
        w(i_trials,:)       = w(i_trials-1,:);
        da(i_trials,:)      = da(i_trials-1,:);
        traj_sim.mu         = mu;
        traj_sim.sa         = 1./pi;
        traj_sim.muhat      = muhat;
        traj_sim.sahat      = 1./pihat;
        continue;
    end

    trial = i_trials - 1;  % 1..n_trials

    % ---------------------------------------------------------------------
    % Determine current contingency probabilities pA,pB,pC
    if strcmpi(species, 'monkey')
        % Predefined monkey schedule
        % (choices y(i_trials) not yet updated; outcome uses y below)
        % p's are implicit in stabl.txt
        pA = NaN; pB = NaN; pC = NaN; %#ok
    else
        % HUMAN: multi-reversal + performance-dependent schedule
        if change_contingency
            % Performance-triggered new row
            if trial <= 80
                prob_row           = randi(size(prob_set1, 1));
                p                  = prob_set1(prob_row, :);
                previous_prob_row1 = prob_row;
            else
                prob_row           = randi(size(prob_set2, 1));
                p                  = prob_set2(prob_row, :);
                previous_prob_row2 = prob_row;
            end
            change_contingency = false;
        else
            % Fixed-schedule reversals
            if trial <= 80
                % "Block 1" (trials 1–80); note dummy => i_trials 2..81
                if i_trials == 2 || i_trials == 41
                    prob_row           = randi(size(prob_set1,1));
                    p                  = prob_set1(prob_row, :);
                    previous_prob_row1 = prob_row;
                else
                    p = prob_set1(previous_prob_row1, :);
                end
            else
                % "Block 2" (trials 81–160); dummy => i_trials 82..161
                if i_trials == 81 || i_trials == 121
                    prob_row           = randi(size(prob_set2,1));
                    p                  = prob_set2(prob_row, :);
                    previous_prob_row2 = prob_row;
                else
                    p = prob_set2(previous_prob_row2, :);
                end
            end
        end

        % Performance-dependent reversal: 9/10 on best deck
        if trial >= 10
            last_10_choices = y(i_trials-9:i_trials);
            [~, best_choice] = max(p);
            if sum(last_10_choices == best_choice) >= 9
                change_contingency = true;
            end
        end

        pA = p(1);
        pB = p(2);
        pC = p(3);
    end

    % ---------------------------------------------------------------------
    % Build infStates for softmax & generate choice
    if i_trials-1 == 1
        % First real trial: use priors
        infStates(1,1,:,1) = muhat(1,1,:);
        infStates(1,3,:,3) = param.p_prc.mu_0(3);
    else
        infStates(i_trials-1,:,:,1) = traj_sim.muhat(i_trials-1,:,:);
        infStates(i_trials-1,:,:,2) = traj_sim.sahat(i_trials-1,:,:);
        infStates(i_trials-1,:,:,3) = traj_sim.mu(i_trials-1,:,:);
        infStates(i_trials-1,:,:,4) = traj_sim.sa(i_trials-1,:,:);
    end

    sim_prob = tapas_softmax_mu3_paella(param, infStates);

    % Choice (1,2,3)
    y(i_trials) = randsample([1 2 3], 1, true, sim_prob(i_trials-1,:));

    % ---------------------------------------------------------------------
    % Generate outcome u(i_trials)
    if strcmpi(species, 'monkey')
        u(i_trials) = table2array(prob_reinforced(i_trials, y(i_trials)));
    else
        p_choice = eval(sprintf('p%c', char('A' + y(i_trials) - 1)));
        u(i_trials) = binornd(1, p_choice);
    end

    % ---------------------------------------------------------------------
    % HGF state update given choice & outcome
    % 2nd level prediction
    muhat(i_trials,2,:) = mu(i_trials-1,2,:) + t(i_trials)*phi(2)*(m(2) - mu(i_trials-1,2,:));

    % 1st level
    muhat(i_trials,1,:) = tapas_sgm(ka(1) * muhat(i_trials,2,:), 1);
    pihat(i_trials,1,:) = 1 ./ (muhat(i_trials,1,:) .* (1 - muhat(i_trials,1,:)));

    pi(i_trials,1,:)    = pihat(i_trials,1,:);
    pi(i_trials,1,y(i_trials)) = Inf;

    mu(i_trials,1,:)          = muhat(i_trials,1,:);
    mu(i_trials,1,y(i_trials))= u(i_trials);

    da(i_trials,1) = mu(i_trials,1,y(i_trials)) - muhat(i_trials,1,y(i_trials));

    % 2nd level precision & update
    pihat(i_trials,2,:) = 1 ./ (1./pi(i_trials-1,2,:) + exp(ka(2)*mu(i_trials-1,3,:) + om(2)));
    pi(i_trials,2,:)    = pihat(i_trials,2,:) + ka(1)^2 ./ pihat(i_trials,1,:);

    mu(i_trials,2,:)          = muhat(i_trials,2,:);
    mu(i_trials,2,y(i_trials))= muhat(i_trials,2,y(i_trials)) + ...
                                ka(1)/pi(i_trials,2,y(i_trials)) * da(i_trials,1);

    da(i_trials,2) = (1/pi(i_trials,2,y(i_trials)) + ...
                      (mu(i_trials,2,y(i_trials)) - muhat(i_trials,2,y(i_trials)))^2) * ...
                     pihat(i_trials,2,y(i_trials)) - 1;

    % Higher levels (if any)
    if n_levels > 3
        for j = 3:(n_levels-1)
            muhat(i_trials,j,:) = mu(i_trials-1,j,:) + t(i_trials)*phi(j)*(m(j) - mu(i_trials-1,j));
            pihat(i_trials,j,:) = 1 ./ (1./pi(i_trials-1,j,:) + ...
                                       t(i_trials)*exp(ka(j)*mu(i_trials-1,j+1,:) + om(j)));

            v(i_trials,j-1) = t(i_trials) * exp(ka(j-1)*mu(i_trials-1,j,y(i_trials)) + om(j-1));
            w(i_trials,j-1) = v(i_trials,j-1) * pihat(i_trials,j-1,y(i_trials));

            pi(i_trials,j,:) = pihat(i_trials,j,:) + ...
                0.5*ka(j-1)^2 * w(i_trials,j-1) .* (w(i_trials,j-1) + (2*w(i_trials,j-1)-1).*da(i_trials,j-1));

            if pi(i_trials,j,1) <= 0
                error('tapas:hgf:NegPostPrec', ...
                      'Negative posterior precision. Parameters are in a region where model assumptions are violated.');
            end

            mu(i_trials,j,:) = muhat(i_trials,j,:) + ...
                0.5 * (1./pi(i_trials,j,:)) .* ka(j-1) .* w(i_trials,j-1) .* da(i_trials,j-1);

            da(i_trials,j) = (1/pi(i_trials,j,y(i_trials)) + ...
                             (mu(i_trials,j,y(i_trials)) - muhat(i_trials,j,y(i_trials)))^2) * ...
                             pihat(i_trials,j,y(i_trials)) - 1;
        end
    end

    % Last level
    muhat(i_trials,n_levels,:) = mu(i_trials-1,n_levels,:) + ...
                                 t(i_trials)*phi(n_levels)*(m(n_levels) - mu(i_trials-1,n_levels));

    pihat(i_trials,n_levels,:) = 1 ./ (1./pi(i_trials-1,n_levels,:) + t(i_trials)*th);

    v(i_trials,n_levels)   = t(i_trials)*th;
    v(i_trials,n_levels-1) = t(i_trials)*exp(ka(n_levels-1)*mu(i_trials-1,n_levels,y(i_trials)) + om(n_levels-1));
    w(i_trials,n_levels-1) = v(i_trials,n_levels-1) * pihat(i_trials,n_levels-1,y(i_trials));

    pi(i_trials,n_levels,:) = pihat(i_trials,n_levels,:) + ...
        0.5 * ka(n_levels-1)^2 * w(i_trials,n_levels-1) .* ...
        (w(i_trials,n_levels-1) + (2*w(i_trials,n_levels-1)-1).*da(i_trials,n_levels-1));

    if pi(i_trials,n_levels,1) <= 0
        error('tapas:hgf:NegPostPrec', ...
              'Negative posterior precision. Parameters are in a region where model assumptions are violated.');
    end

    mu(i_trials,n_levels,:) = muhat(i_trials,n_levels,:) + ...
        0.5 * (1./pi(i_trials,n_levels,:)) .* ka(n_levels-1) .* ...
        w(i_trials,n_levels-1) .* da(i_trials,n_levels-1);

    da(i_trials,n_levels) = (1/pi(i_trials,n_levels,y(i_trials)) + ...
                            (mu(i_trials,n_levels,y(i_trials)) - muhat(i_trials,n_levels,y(i_trials)))^2) * ...
                            pihat(i_trials,n_levels,y(i_trials)) - 1;

    traj_sim.mu    = mu;
    traj_sim.sa    = 1./pi;
    traj_sim.muhat = muhat;
    traj_sim.sahat = 1./pihat;

    % Coupled updating (2-bandit special case)
    if coupled
        if y(i_trials) == 1
            mu(i_trials,1,2) = 1 - mu(i_trials,1,1);
            mu(i_trials,2,2) = tapas_logit(1 - tapas_sgm(mu(i_trials,2,1),1), 1);
        elseif y(i_trials) == 2
            mu(i_trials,1,1) = 1 - mu(i_trials,1,2);
            mu(i_trials,2,1) = tapas_logit(1 - tapas_sgm(mu(i_trials,2,2),1), 1);
        end
    end
end
end
