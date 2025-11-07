%% Decoding script:
%   - Decode own & opponent's response for current & previous trial
%
% Toolboxes needed: fieldtrip (we used version 20240110 here), cosmomvpa
%
% Notes:
%   - We excluded pair 10 (major CMS issues for ppt 2), 23 (no triggers),
%   and 24 (major CMS issues for ppt 2 - first 32 trials only)

clearvars; clc;

%% Set the path
path_to_data = '../data';

%% Set parameters
pair_ids = [1:9,11:22,25:34];   % Pair IDs (Pair 10 (major CMS issues for ppt 2), 23 (no triggers), 24 (major CMS issues for ppt 2 for first first 32 trials) were excluded)
num_pairs = size(pair_ids,2);   % Number of pairs
num_trials = 480;
num_chan = 64;

%% Loop over pairs
% Pre-allocate the output
decoding_accuracy = cell(1,4);
searchlight_acc = cell(1,4);
pair_idx = reshape(1:num_pairs*2,[2,num_pairs])';

for p = 1:num_pairs

    % Get the pair ID
    pair = pair_ids(1,p);
    fprintf('Loading pair %.0f of %.0f\n',p,num_pairs);

    % Load in the behaviour
    events = readtable(fullfile(path_to_data,num2str(pair,'sub-%02d'),'eeg',num2str(pair,'sub-%02d_task-RPS_events.tsv')),'Filetype','text','delimiter','\t');

    % We need the following columns:
    % Column 5 - Player 1 played: 1) Rock 2) Paper 3) Scissors
    % Column 7 - Player 2 played: 1) Rock 2) Paper 3) Scissors
    % Column 9 - Outcome: 1) draw, 2) player 1 wins, 3) player 2 wins
    events = events(:,[5,7,9]);

    % Get the behavioural responses in the right format
    % Column 1 - This player played: 1) Rock 2) Paper 3) Scissors
    % Column 2 - Other player played: 1) Rock 2) Paper 3) Scissors
    % Column 3 - Outcome: 1) draw, 2) this wins, 3) other player wins
    % Column 4 - In the previous trial, this player played: 1) Rock 2) Paper 3) Scissors
    % Column 5 - In the previous trial, the other player played: 1) Rock 2) Paper 3) Scissors
    % Get the data for player 1
    Player_1_Behav = [table2array(events),[nan,nan;table2array(events(1:end-1,1:2))]];
    % Get the data for player 2
    Player_2_Behav = [table2array(events(:,[2,1])),zeros(num_trials,1),[nan,nan;table2array(events(1:end-1,[2,1]))]];
    % Change coding of column 3 for player 2 (outcome) to code outcome relative to player 2
    Player_2_Behav(Player_1_Behav(:,3)==1,3) = 1;
    Player_2_Behav(Player_1_Behav(:,3)==2,3) = 3;
    Player_2_Behav(Player_1_Behav(:,3)==3,3) = 2;
    all_behav_data = cat(3,Player_1_Behav,Player_2_Behav);

    % Set random seed
    rng(p);

    % Loop over the 2 players in the pair
    for ppt = 1:2

        % Load the pre-processed EEG data
        fprintf('   ppt %.0f\n',ppt);
        load(sprintf('%s/derivatives/pair-%02d_player-%01d_task-RPS_eeg.mat',path_to_data,pair,ppt));

        % Re-reference to the average reference
        cfg=[];
        cfg.reref      = 'yes';
        cfg.refchannel = 1:64;
        eeg_data = ft_preprocessing(cfg,eeg_data);

        % Split the data into 3 parts:
        %  - Get ready (2s)
        %  - Response (2s)
        %  - Feeback (1s)
        % Do separate baseline corrections for each part
        cfg = [];
        cfg.latency = [-0.2 2];
        eeg_data_partA = ft_selectdata(cfg,eeg_data);
        cfg.latency = [1.8 4];
        eeg_data_partB = ft_selectdata(cfg,eeg_data);
        cfg.latency = [3.8 5];
        eeg_data_partC = ft_selectdata(cfg,eeg_data);

        % Shift the time labels for part B and C to make 0 the start of the
        % response (B) or start of the feedback (C)
        for trial_num = 1:num_trials
            eeg_data_partB.time{trial_num} = eeg_data_partA.time{trial_num};
            eeg_data_partC.time{trial_num} = eeg_data_partA.time{trial_num}(1,1:length(eeg_data_partC.time{trial_num}));
        end

        % Baseline-correction: use the [-0.2 0] as a baseline.
        % Run the baseline corrections for the trial parts
        cfg = [];
        cfg.demean = 'yes';
        cfg.baselinewindow = [-0.2 0];
        eeg_data_partA = ft_preprocessing(cfg,eeg_data_partA);
        eeg_data_partB = ft_preprocessing(cfg,eeg_data_partB);
        eeg_data_partC = ft_preprocessing(cfg,eeg_data_partC);

        % Get the behavioural responses
        behav_data = all_behav_data(:,:,ppt);

        % We can't decode the previous trial for the first trial of each
        % block, because we don't have a history for this trial yet.
        % We remove the first trial of each block (from behavioural data as
        % well as EEG)
        rem_idx = 1:40:480;
        behav_data(rem_idx,:) = [];

        % Do this is the EEG data as well as the behavioural data
        sel_idx = 1:num_trials;
        sel_idx(rem_idx) = [];
        cfg = [];
        cfg.trials = sel_idx;
        eeg_data_partA = ft_selectdata(cfg,eeg_data_partA);
        eeg_data_partB = ft_selectdata(cfg,eeg_data_partB);
        eeg_data_partC = ft_selectdata(cfg,eeg_data_partC);

        %% Average the data into time bins, and re-combine into 1 dataset
        % (rather than 3 parts)
        time_windows_AB = [0:0.25:1.75;0.25:0.25:2]';   % 0 to 2 seconds
        time_windows_C = [0:0.25:0.75;0.25:0.25:1]';    % 0 to 1 second

        % Pre-allocate the output dataset
        eeg_data = eeg_data_partA;
        eeg_data.trial = [];
        eeg_data.time = [];

        % Loop over trials
        for trial_num = 1:size(eeg_data_partA.trial,2)
            % Pre-allocate temporary matrix for the tial data
            % We do this separately for A (decision) and B (response)
            temp_A = zeros(num_chan,size(time_windows_AB,1));
            temp_B = zeros(num_chan,size(time_windows_AB,1));
            % Loop over the time bins for this part. Get the data for
            % time-points in this time bin and average.
            for w_idx = 1:size(time_windows_AB,1)
                temp_A(:,w_idx) = mean(eeg_data_partA.trial{trial_num}(:,eeg_data_partA.time{trial_num}>time_windows_AB(w_idx,1)&eeg_data_partA.time{trial_num}<time_windows_AB(w_idx,2)),2);
                temp_B(:,w_idx) = mean(eeg_data_partB.trial{trial_num}(:,eeg_data_partB.time{trial_num}>time_windows_AB(w_idx,1)&eeg_data_partB.time{trial_num}<time_windows_AB(w_idx,2)),2);
            end
            % Do the same for part C (feeback)
            temp_C = zeros(num_chan,size(time_windows_C,1));
            for w_idx = 1:size(time_windows_C,1)
                temp_C(:,w_idx) = mean(eeg_data_partC.trial{trial_num}(:,eeg_data_partC.time{trial_num}>time_windows_C(w_idx,1)&eeg_data_partC.time{trial_num}<time_windows_C(w_idx,2)),2);
            end

            % Now we can add the data to the the big matrix
            eeg_data.trial{trial_num} = [temp_A,temp_B,temp_C];
            eeg_data.time{trial_num} = [time_windows_AB(:,2);time_windows_AB(:,2)+2;time_windows_C(:,2)+4]';

        end

        %% Convert the data from fieldtrip to cosmomvpa format
        cfg = [];
        cfg.keeptrials = 'yes';
        ds = cosmo_meeg_dataset(ft_timelockanalysis(cfg,eeg_data));
        ds.sa = table2struct(array2table(behav_data,'VariableNames',{'self','other','result','selfp','otherp'}),'toscalar',1);

        %% Run the decoding
        % Loop over things we want to decode
        % 1 = played self
        % 2 = played other
        % 3 = played self previous trial
        % 4 = played other previous trial
        test_idx = 1:4;
        for test = 1:size(test_idx,2)

            %%% DECODING %%%
            % Save dataset under a new name
            ds_sel = ds;

            % Check what we decode and set the targets (what we decode)
            switch test_idx(test)
                case 1
                    ds_sel.sa.targets = ds_sel.sa.self;
                case 2
                    ds_sel.sa.targets = ds_sel.sa.other;
                case 3
                    ds_sel.sa.targets = ds_sel.sa.selfp;
                case 4
                    ds_sel.sa.targets = ds_sel.sa.otherp;
            end

            % remove no-responses
            ds_sel = cosmo_slice(ds_sel,ds_sel.sa.targets>0);

            % Make 10 chunks that are as balanced as possible based on targets
            ds_sel.sa.chunks = (1:numel(ds_sel.sa.targets))';
            ds_sel.sa.chunks = cosmo_chunkize(ds_sel,10);

            % Average sample to improve signal to noise ratio. This also
            % fixes the different number of trials for each response. We
            % average 4 random samples together and repeat this 20 times.
            ds_sel = cosmo_average_samples(ds_sel,'count',4,'repeats',20,'seed',1);

            % define the neighbourhood (individual timepoints)
            nh = cosmo_interval_neighborhood(ds_sel,'time','radius',0);

            % classification parameters
            measure = @cosmo_crossvalidation_measure;
            ma = {};
            ma.partitions = cosmo_nfold_partitioner(ds_sel);
            % use LDA
            ma.classifier = @cosmo_classify_lda;
            % optional: use multiple cores in parallel
            ma.nproc = 1;
            ma.output = 'accuracy';

            % Run the decoding
            res = cosmo_searchlight(ds_sel,nh,measure,ma);

            % Add additional information to the output
            res.sa.pair = pair;
            res.sa.player = ppt;
                
            % Save the decoding accuracy
            decoding_accuracy{test} = res;

            %%% CHANNEL SEARCHLIGHT %%%
            % Make the neighbourhoors - set up channel neighbours for the searchlight
            % Select 4 (or 5) neighbours
            nh1 = cosmo_meeg_chan_neighborhood(ds_sel,'count',4,'label','dataset','label_threshold',.99);
            nh2 = cosmo_interval_neighborhood(ds_sel,'time','radius',0);
            nh_sl = cosmo_cross_neighborhood(ds_sel,{nh1,nh2});

            % Run the searchlight
            res_sl = cosmo_searchlight(ds_sel,nh_sl,measure,ma);

            % Add additional information to the output
            res_sl.sa.pair = pair;
            res_sl.sa.player = ppt;
                
            % Save the decoding accuracy
            searchlight_acc{test} = res_sl;

        end % Loop over the things to decode

        % Save the decoding & searchlight results for this player
        save(sprintf('%s/derivatives/pair-%02d_player-%01d_task-RPS_decoding.mat',path_to_data,pair,ppt),'decoding_accuracy','searchlight_acc');

    end % Loop over the 2 players in the pair
end % Loop over pairs


