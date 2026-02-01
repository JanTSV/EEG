%% RPS markov chain
% 
% Predict the response of the player based on N previous trials. We use the
% accuracy as a measure of predictability of the player. We use different
% window sizes of 5 - 100 previous trials.
%
% Toolboxes needed: n/a
%
% Notes:
%   - We excluded pair 10 (major CMS issues for ppt 2), 23 (no triggers),
%   and 24 (major CMS issues for ppt 2 - first 32 trials only)

clearvars; clc;

%% Set the path
path_to_data = '/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download'; %'../data';

%% Set parameters
pair_ids = [1:9,11:22,25:34];   % Pair IDs (Pair 10 (major CMS issues for ppt 2), 23 (no triggers), and 24 (major CMS issues for ppt 2 - first 32 trials only) were excluded)
num_pairs = size(pair_ids,2);   % Number of pairs
num_trials = 480;
num_windows = 100;

% Pre-allocate the putput
% Mean_Accuracy: 31 pairs x 2 players x n windows - overall accuracy of the Markov chain
Mean_Accuracy = zeros(num_pairs,2,num_windows);
% M_pred: 31 pairs x 2 players x n windows x 480 trial x 4 columns
% specifying:
%   - Column 1: actual response of this ppt (R/P/S)
%   - Column 2: response predicted based on N previous trials. We vary N (window size).
%   - Column 3: probability of the predicted response (is there a pattern in previous responses, or were they random)
%   - Column 4: is the predicted response the same as the actual response?
M_pred = zeros(num_pairs,2,num_windows,num_trials,4);

% Load the behavioural data
for p = 1:num_pairs
    % Get the pair ID
    pair = pair_ids(1,p);
    fprintf('Loading pair %.0f of %.0f\n',p,num_pairs);
    
    % Loop over the 2 ppts in the pair
    for ppt = 1:2

        % Load in the behaviour
        events = readtable(fullfile(path_to_data,num2str(pair,'sub-%02d'),'eeg',num2str(pair,'sub-%02d_task-RPS_events.tsv')),'Filetype','text','delimiter','\t');
        
        % Get the response for this ppt for each trial 
        ppt_resp =  [events.player1_resp,events.player2_resp];
        resp = ppt_resp(:,ppt);

        % Pre-allocate the 'prob_data' matrix that gives an overview of how 
        % frequently each move is played, and how frequently which moves follows which
        % Column 1:  Trial
        % Column 2:  N_Rock
        % Column 3:  R_R
        % Column 4:  R_P
        % Column 5:  R_S
        % Column 6:  N_Paper
        % Column 7:  P_R
        % Column 8:  P_P
        % Column 9:  P_S
        % Column 10: N_Scissors
        % Column 11: S_R
        % Column 12: S_P
        % Column 13: S_S
        prob_data = NaN(num_trials,13);
        % Add the data for the first trial
        prob_data(1,:) = [1,3,1,1,1,3,1,1,1,3,1,1,1];

        % Loop over the remaining trials in the experiment
        for i = 2:num_trials

            % Add the data from the previous trial to the current trial
            prob_data(i,:) = prob_data(i-1,:);

            % Update the trial number
            prob_data(i,1) = i;

            % Update the count based on the current trial
            if resp(i-1)==1 % If previous response is rock
                prob_data(i,2)=prob_data(i-1,2)+1;

                % If current response is:
                if resp(i)==1 % R
                    prob_data(i,3)=prob_data(i-1,3)+1;
                elseif resp(i)==2 % P
                    prob_data(i,4)=prob_data(i-1,4)+1;
                else % S
                    prob_data(i,5)=prob_data(i-1,5)+1;
                end

            elseif resp(i-1)==2 % If previous response is paper
                prob_data(i,6)=prob_data(i-1,6)+1;
                
                % If current response is:
                if resp(i)==1 %
                    prob_data(i,7)=prob_data(i-1,7)+1;
                elseif resp(i)==2 % P
                    prob_data(i,8)=prob_data(i-1,8)+1;
                else % S
                    prob_data(i,9)=prob_data(i-1,9)+1;
                end

            elseif resp(i-1)==3 % If previous response is scissors
                prob_data(i,10)=prob_data(i-1,10)+1;
                
                % If current response is:
                if resp(i)==1 % R
                    prob_data(i,11)=prob_data(i-1,11)+1;
                elseif resp(i)==2 % P
                    prob_data(i,12)=prob_data(i-1,12)+1;
                else % S
                    prob_data(i,13)=prob_data(i-1,13)+1;
                end
            end

        end % Loop over trials

        %%%%% Make the probability matrix:
        %%%"played","predicted","probability","accuracy"
        %   - Column 1: actual response of this ppt (R/P/S)
        %   - Column 2: response predicted based on N previous trials. We vary N (window size).
        %   - Column 3: probability of the predicted response (is there a pattern in previous responses, or were they random)
        %   - Column 4: is the predicted response the same as the actual response?
        
        % Pre-allocate the probability matrix
        prob_res = NaN(num_trials,4);
        % Set the start probability (1/3) for all options of two consecutive
        % responses
        m_Prob = [1/3,1/3,1/3;1/3,1/3,1/3;1/3,1/3,1/3];

        inter_prob_data = NaN(num_trials,13);

        % We use different window sizes (previous N trials used)
        for window_size = 5:100
            for i = 3:num_trials % Loop over trials

                % Get the additional responses for this window
                if i<window_size+1
                    % If we don't have the required window size yet, just
                    % get the change so far in this experiment 
                    inter_prob_data(i,:) = prob_data(i-1,:);
                else
                    % Get the change in this window 
                    inter_prob_data(i,:) = prob_data(i-1,:)-prob_data(i-window_size,:);
                end

                % Update the trial
                inter_prob_data(i,1)=i;
            
                % In this window, for each combination of 2 consecutive
                % responses, check the proportion this occurred
                % If there have been rock responses
                if inter_prob_data(i,2)>0
                    % Get the proportion that was preceded by each other response
                    m_Prob(1,:)=[inter_prob_data(i,3)/inter_prob_data(i,2),inter_prob_data(i,4)/inter_prob_data(i,2),inter_prob_data(i,5)/inter_prob_data(i,2)];
                else
                    m_Prob(1,:)=[1/3,1/3,1/3];
                end
                % If there have been paper responses
                if inter_prob_data(i,6)>0
                    % Get the proportion that was preceded by each other 
                    m_Prob(2,:)=[inter_prob_data(i,7)/inter_prob_data(i,6),inter_prob_data(i,8)/inter_prob_data(i,6),inter_prob_data(i,9)/inter_prob_data(i,6)];
                else
                    m_Prob(2,:)=[1/3,1/3,1/3];
                end
                % If there have been scissors responses
                if inter_prob_data(i,10)>0
                    % Get the proportion that was preceded by each other 
                    m_Prob(3,:)=[inter_prob_data(i,11)/inter_prob_data(i,10),inter_prob_data(i,12)/inter_prob_data(i,10),inter_prob_data(i,13)/inter_prob_data(i,10)];
                else
                    m_Prob(3,:)=[1/3,1/3,1/3];
                end

                %%% Threshold 1/3 probability for decoding
                % Make the prob_res matrix. Add column 1: actual response
                prob_res(i,1)=resp(i);

                % It is possible the ppt did not respond on the last trial. 
                % If this happened, we just use the response from the trial 
                % before (or the one before that)
                if resp(i-1)>0
                    idx=i;
                elseif resp(i-2)>0 % If the ppt did not respond on the previous trial
                    idx=i-1; %%% accounts for only one missing response
                else % If the ppt did not respond on the previous 2 trials
                    idx=i-2; %%% accounts for only one missing response
                end
                % Get the last response and based on that, get the most
                % likely next response & proabability.
                if idx>1
                    if resp(idx-1)==1 % If last response was rock
                        idx_max=find(m_Prob(1,:)==max(m_Prob(1,:)));
                        prob_res(i,2)=idx_max(1); % Get the most likely next response
                        prob_res(i,3)=max(m_Prob(1,:)); % Get the probability
                    elseif resp(idx-1)==2 % If last response was paper
                        idx_max=find(m_Prob(2,:)==max(m_Prob(2,:)));
                        prob_res(i,2)=idx_max(1); % Get the most likely next response
                        prob_res(i,3)=max(m_Prob(2,:)); % Get the probability
                    elseif resp(idx-1)==3 % If last response was scissors
                        idx_max=find(m_Prob(3,:)==max(m_Prob(3,:)));
                        prob_res(i,2)=idx_max(1); % Get the most likely next response
                        prob_res(i,3)=max(m_Prob(3,:)); % Get the probability
                    end
                end
                % Check whether the prediction (most likely next response)
                % was accurate or not
                if isnan(prob_res(i,3)) % Deal with NaN values
                    prob_res(i,4)=NaN;
                elseif prob_res(i,1)==prob_res(i,2)
                    prob_res(i,4)=1;
                else
                    prob_res(i,4)=0;
                end

                % Save the accuracy of our Markov chain
                data_mean = prob_res(3:480,4);
                data_mean = data_mean(isfinite(data_mean));
                Mean_Accuracy(p,ppt,window_size) = mean(data_mean);

                % Save the probability/prediction matrix
                M_pred(p,ppt,window_size,:,:)= prob_res;

            end % Loop over trials
        end % Loop over window size


    end % Loop over participant 1/2 in the pair
end % Loop over pairs


%% Save the output
save(fullfile(path_to_data,'derivatives','markov_chain_pred.mat'),'M_pred','Mean_Accuracy');




