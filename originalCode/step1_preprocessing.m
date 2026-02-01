%% Pre-processing script:
%   - Plot the data so we can identify noisy channels
%   - Interpolate noisy channels
%   - Down-sample to 256 Hz (to make the data easier to work with)
%   - Save
%
% Toolboxes needed: fieldtrip (we used version 20240110 here)
%
% Notes:
%   - The 2 pairs are in a single file in the raw data. We have to select
%   different channels based on the pair number.

%% Set the path
clear; close all; clc
addpath(genpath(fullfile(pwd,'helperFiles')));
ft_defaults;
 
path_to_data =  '/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download'; %'../data';
path_to_code = '../code';

%% Set parameters
% If identify_bad_channels is true, we plot the data so we can identify bad 
% channels. Skip if we've already identified bad channels and want to fix them. 
% To fix bad channels, set interpolate_bad_channels to true. This assumes we
% have a tsv file called 'participants.tsv' with labels
% of the bad channels.
identify_bad_channels = false;
interpolate_bad_channels = true;

% Set parameters
num_trials = 480;               % There were 480 games (trials) in the experiment
pair_ids = [1:9,11:22,25:34];   % Pair IDs (Pair 10 (major CMS issues for ppt 2), 23 (no triggers), 24 (major CMS issues for ppt 2 for first first 32 trials) were excluded)
num_pairs = size(pair_ids,2); % Number of pairs
FS = 2048;                      % Biosemi sampling frequency

% Load the demographics - this has information about which channels are bad
participants = readtable(fullfile(path_to_data, 'participants.tsv'),'FileType','text','Delimiter','\t');
% show whats inside
head(participants)
participants.Properties.VariableNames

%% Run pre-processing 
% Load the data. If we want to find bad channels, plot the data for visual 
% inspection. Interpolate bad channels (found in demographics file), down-sample 
% to 256 Hz and save.

% Loop over pairs
fprintf("Debug - loop starting ...")
for p = 1:num_pairs
    % Get the pair ID
    fprintf("Debug - inside loop")
    pair = pair_ids(1,p);
    fprintf('Loading pair %.0f of %.0f\n',p,num_pairs);

    % Get the trigger times (for the start of each trial)
    events_filename = fullfile(path_to_data,num2str(pair,'sub-%02d'),'eeg',num2str(pair,'sub-%02d_task-RPS_events.tsv'));
    events = readtable(events_filename,'FileType','text','Delimiter','\t');
    stimonsample = events.onset_sample;

    % Specify the epoch: -0.2 to 5 sec (relative to onset of 'Decision' screen)
    prestim = 0.2;  % epoch start
    poststim = 5; % epoch end
    
    % Make trial matrix
    TRL = [stimonsample-ceil(prestim*FS),ceil(stimonsample+poststim*FS)];
    TRL(:,3) = TRL(:,1)-stimonsample;

    % Read the header (to get the channel labels
    raw_filename = fullfile(path_to_data,num2str(pair,'sub-%02d'),'eeg',num2str(pair,'sub-%02d_task-RPS_eeg.bdf'));
    hdr = ft_read_header(raw_filename);

    % Loop over player 1 and 2 in the pair
    for ppt = 1:2 

        % There are some inconsistencies about whether additional (not
        % used) channels are recorded or not. We'll work out which channels
        % we need based on the labels. Also note that who is named player 1
        % and 2 in the EEG data does not match the behavioural data (+ demographics). 
        % We'll swap this around here to correct for this!
        % We have:
        %   - Player 1: 2-A1 to 2-A32 & 2-B1 to 2-B32
        %   - Player 2: 1-A1 to 1-A32 & 1-B1 to 1-B32
        chan_idx = [contains(hdr.label,'2-A')+contains(hdr.label,'2-B'),contains(hdr.label,'1-A')+contains(hdr.label,'1-B')];
        orig_label = hdr.label(chan_idx(:,ppt)==1);

        % Load and epoch the data, 
        cfg = [];
        cfg.datafile = raw_filename;
        cfg.trl = TRL;
        cfg.channel = orig_label;
        data_epoch = ft_preprocessing(cfg);

        % Add the correct channel names
        layout = ft_prepare_layout(struct('layout','biosemi64.lay'));
        data_epoch.label(1:64) = layout.label(1:64);

        % Label the structure of the data
        data_epoch.dimord = 'chan_time';

        % Do we want to plot the data to identify bad channels?
        if identify_bad_channels
    
            % Highpass filter (for plotting purposes only)
            cfg = [];
            cfg.hpfilter   = 'yes';
            cfg.hpfreq     = 0.1;
            cfg.hpfiltord  = 4;
            cfg.hpfilttype = 'but'; %default
            cfg.hpfiltdir  = 'twopass'; %default
            cfg.lpfilter   = 'yes';
            cfg.lpfreq     = 100;
            cfg.lpfiltord  = 4;
            cfg.lpfilttype = 'but'; %default
            cfg.lpfiltdir  = 'twopass'; %default
            data_epoch_f = ft_preprocessing(cfg,data_epoch);

            % Visualise the data so we can identify noisy channels
            cfg=[];
            cfg.layout = 'biosemi64.lay';
            cfg.highlight = 'labels';        
            cfg.preproc.dftfilter = 'no';
            cfg.continuous   = 'no'; 
            cfg.viewmode = 'vertical'; 
            cfg.ylim = [-100 100]; 
            cfg.allowoverlap = 'yes';
            cfg.plotlabels = 'yes';
            ft_databrowser(cfg,data_epoch_f);
        end
    
        % Do we want to interpolate the bad channels?
        if interpolate_bad_channels
            fprintf("\nDebug - inside interpolate bad channels if clause\n")

            % Get the channels to fix for this ppt (this information is in the
            % participants.tsv file).
            chan_to_fix = participants(strcmp(participants.participant_id,num2str(pair,'sub-%02d')),[6,10]); % participants(strcmp(participants.participant_id,num2str(pair,'sub-%02d')),[7,12]);
            chan_to_fix = table2cell(chan_to_fix(1,ppt));
            if ~isempty(chan_to_fix{1}) % Check if there are channels to fix

                % Prepare channels neighbours to fix bad channels
                load('biosemi64.mat');          % Loading 3D template 
                elec = [];
                elec.pnt = biosemi64;           % The 3D-Positions
                elec.label = data_epoch.label;  % and the names
                cfg = [];
                cfg.method = 'distance';        % Select the neighbours by distance
                cfg.neighbourdist = .5;
                cfg.elec = elec;
                neighbours = ft_prepare_neighbours(cfg);
    
                % Fix the bad channels
                cfg = [];
                cfg.badchannel = strsplit(chan_to_fix{1},', ');
                cfg.neighbours = neighbours; 
                cfg.elec = elec;
                data_epoch = ft_channelrepair(cfg,data_epoch);

                % Update
                fprintf('pair %.0f, player %.0f: fixed %s\n',pair,ppt,chan_to_fix{1});
            end % Check if there are channels to fix

            % Down-sample the data to 256 Hz (to make it easier to work with)
            cfg = [];
            cfg.resamplefs = 256;
            eeg_data = ft_resampledata(cfg,data_epoch);
    
            % Save the epoched data (noisy channels fixed and down-sampled, no filtering)
            save(sprintf('%s/derivatives/pair-%02d_player-%01d_task-RPS_eeg.mat',path_to_data,pair,ppt),'eeg_data','-V7.3');

            % Clear old data struct
            data_epoch = [];
            eeg_data = [];

        end % Do we want to interpolate the bad channels?
    end % Loop over the 2 ppts in the pair
end % Loop over pairs