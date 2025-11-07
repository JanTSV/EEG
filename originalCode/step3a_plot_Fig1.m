%% Plot the behavioural responses
%
% Toolboxes needed: 
% daviolinplot: https://www.mathworks.com/matlabcentral/fileexchange/136524-daviolinplot-beautiful-violin-and-raincloud-plots
%
% Notes:
%   - We excluded pair 10 (major CMS issues for ppt 2), 23 (no triggers),
%   and 24 (major CMS issues for ppt 2 - first 32 trials only)

clearvars; clc;

%% Set the path
path_to_data = '../data';
if ~exist(fullfile(path_to_data,'derivatives','plots'),'dir')
    mkdir(fullfile(path_to_data,'derivatives'),'plots');
end

%% Set parameters
pair_ids = [1:9,11:22,25:34];   % Pair IDs (Pair 10 (major CMS issues for ppt 2), 23 (no triggers), and 24 (major CMS issues for ppt 2 - first 32 trials only) were excluded)
num_pairs = size(pair_ids,2);   % Number of pairs
pair_idx = reshape(1:num_pairs*2,[2,num_pairs])';
num_blocks = 12;
num_trials_per_block = 40;
num_trials = num_blocks*num_trials_per_block;

%% Loop over pairs
outcome_summary = zeros(num_pairs,3);
all_played_rank = zeros(3,num_pairs*2);
ranked_resp = zeros(3,num_pairs*2);
prop_stay = zeros(3,num_pairs*2);
for p = 1:num_pairs
     
    % Get the pair ID
    pair = pair_ids(1,p);

    % Load in the behaviour
    events = readtable(fullfile(path_to_data,num2str(pair,'sub-%02d'),'eeg',num2str(pair,'sub-%02d_task-RPS_events.tsv')),'Filetype','text','delimiter','\t');

    % Pick the overall winner (based on all trials)
    winner_idx = find([sum(events.outcome==2)>sum(events.outcome==3),sum(events.outcome==2)<sum(events.outcome==3),sum(events.outcome==2)==sum(events.outcome==3)]);
    if winner_idx==3
        warning('There is no winner for this pair');
    end

    % Remove trials where one of the players did not respond
    events_r = events(events.player1_resp>0&events.player2_resp>0,:);

    % Get the outcome of the games
    % Column 1: proportion draw
    % Column 2: proportion winner wins
    % Column 3: proportion loser wins
    outcome_summary(p,:) = ([sum(events_r.outcome==1),sum(events_r.outcome==2),sum(events_r.outcome==3)]/size(events_r,1))*100;
    outcome_summary(p,:) = outcome_summary(p,[1,winner_idx+1,find(~ismember(1:2,winner_idx))+1]); % Sort according to winner/loser

    % Get the proportion of R/P/S responses
    all_played = zeros(3,2);
    played = [events_r.player1_resp,events_r.player2_resp];
    for ppt  = 1:2
        temp = tabulate(played(:,ppt));
        % Column 1 = player 1, Column 2 = pplayer 2
        % Row 1 = rock, Row 2 = paper, Row 2 = scissors
        all_played(:,ppt) = temp(:,3);

        % Sort by rank
        % Row 1 = most played, Row 2 = mid, Row 2 = least played
        % The number shows the response (1 = R, 2 = P, 3 = S)
        [~,idx] = sort(temp(:,3),'descend');
        all_played_rank(:,pair_idx(p,ppt)) = idx;
    end

    % Get the proportion of most/mid/least played, regardless of the
    % response chosen
    temp = sort(all_played,'descend');
    ranked_resp(:,pair_idx(p,1)) = temp(:,1);
    ranked_resp(:,pair_idx(p,2)) = temp(:,2);

    % Get the proportion of 'stay' responses (same response twice), split
    % by game outcome. We do this per block, as there is no previous
    % trial for the first game in each block. 

    % Get the behavioural responses in the right format
    % Column 1 - This player played: 1) Rock 2) Paper 3) Scissors
    % Column 2 - Other player played: 1) Rock 2) Paper 3) Scissors
    % Column 3 - Outcome: 1) draw, 2) this wins, 3) other player wins
    % Column 4 - In the previous trial, this player played: 1) Rock 2) Paper 3) Scissors
    % Column 5 - In the previous trial, the other player played: 1) Rock 2) Paper 3) Scissors
    % Get the data for player 1
    Player_1_Behav = table2array(events(:,[5,7,9]));
    % Get the data for player 2
    Player_2_Behav = [table2array(events(:,[7,5])),zeros(size(events,1),1)];
    % Change coding of column 3 for player 2 (outcome) to code outcome relative to player 2
    Player_2_Behav(Player_1_Behav(:,3)==1,3) = 1;
    Player_2_Behav(Player_1_Behav(:,3)==2,3) = 3;
    Player_2_Behav(Player_1_Behav(:,3)==3,3) = 2;

    % Reshape into blocks - we used the original response data here
    % (without null responses removed), so we don't have to deal with
    % missing trials. We deal with null response trial below.
    Player_1_Behav = permute(reshape(Player_1_Behav,[num_trials_per_block,num_blocks,3]),[1,3,2]);
    Player_2_Behav = permute(reshape(Player_2_Behav,[num_trials_per_block,num_blocks,3]),[1,3,2]);

    p1_draw = [];
    p1_win = [];
    p1_lose = [];
    p2_draw = [];
    p2_win = [];
    p2_lose = [];
    % Loop over blocks
    for block_num = 1:num_blocks
        % Loop over trials (skip the first, as there was no previous response)
        for trial_num = 2:num_trials_per_block
            % Skip if the current or previous trial were no-response trials
            if all([Player_1_Behav(trial_num-1,1,block_num),Player_1_Behav(trial_num,1,block_num),Player_2_Behav(trial_num-1,1,block_num),Player_2_Behav(trial_num,1,block_num)]>0)
                % Player 1
                switch Player_1_Behav(trial_num-1,3,block_num)
                    case 1 % Previous outcome = draw
                        p1_draw = [p1_draw;Player_1_Behav(trial_num,1,block_num)==Player_1_Behav(trial_num-1,1,block_num)];
                    case 2 % Previous outcome = win
                        p1_win = [p1_win;Player_1_Behav(trial_num,1,block_num)==Player_1_Behav(trial_num-1,1,block_num)];
                    case 3 % Previous outcome = lose
                        p1_lose = [p1_lose;Player_1_Behav(trial_num,1,block_num)==Player_1_Behav(trial_num-1,1,block_num)];
                end
    
                % Player 2
                switch Player_2_Behav(trial_num-1,3,block_num)
                    case 1 % Previous outcome = draw
                        p2_draw = [p2_draw;Player_2_Behav(trial_num,1,block_num)==Player_2_Behav(trial_num-1,1,block_num)];
                    case 2 % Previous outcome = win
                        p2_win = [p2_win;Player_2_Behav(trial_num,1,block_num)==Player_2_Behav(trial_num-1,1,block_num)];
                    case 3 % Previous outcome = lose
                        p2_lose = [p2_lose;Player_2_Behav(trial_num,1,block_num)==Player_2_Behav(trial_num-1,1,block_num)];
                end
            end
        end % Loop over trials
    end % Loop over blocks

    % Summarise the strategy 
    prop_stay(1,pair_idx(p,1)) = (sum(p1_win)/size(p1_win,1))*100;
    prop_stay(2,pair_idx(p,1)) = (sum(p1_lose)/size(p1_lose,1))*100;
    prop_stay(3,pair_idx(p,1)) = (sum(p1_draw)/size(p1_draw,1))*100;
    prop_stay(1,pair_idx(p,2)) = (sum(p2_win)/size(p2_win,1))*100;
    prop_stay(2,pair_idx(p,2)) = (sum(p2_lose)/size(p2_lose,1))*100;
    prop_stay(3,pair_idx(p,2)) = (sum(p2_draw)/size(p2_draw,1))*100;

end

%% Markov chain - predictability
% Load the Markov chain data 
load(fullfile(path_to_data,'derivatives','markov_chain_pred.mat'))

% Get the accuracy across all players
pred_acc = [squeeze(Mean_Accuracy(:,1,:));squeeze(Mean_Accuracy(:,2,:))]*100; % convert to percentages
pred_acc = pred_acc(:,5:100); % There is no data for these small window sizes, remove this

% Calculate the 95% confidence interval
CI_pred_acc = (std(pred_acc)/sqrt(size(pred_acc,1))).*repmat(tinv(0.975,size(pred_acc,1)-1),[size(pred_acc,2),1])';

%% Plot
rng(1); % Set seed so that the (individual ppt) dots are always in the same place on the x-axis
font_size = 19;
fh = figure(1);clf;
fh.Position = [100,100,800,800];

%%% C) GAME OUTCOMES %%%
ax3 = axes('Position',[0.066,0.565,0.41,0.385]);
hold on
plot([0,4],[(1/3),(1/3)]*100,'k','LineWidth',1,'LineStyle','--');

% We used these colourmaps: https://www.mathworks.com/matlabcentral/fileexchange/120088-200-colormap
% Switching to standard colourmaps here.
% Make the colourmap
p_col = parula(20);
p_col = p_col([8,5,2],:);

% Plot
h = daviolinplot([outcome_summary(:,2);outcome_summary(:,3);outcome_summary(:,1)],'groups',sort(repmat(1:3,[1,size(outcome_summary,1)]))',...
    'xtlabels', {''},'color',p_col,'scatter',2,'jitter',1,...
    'box',3,'boxcolors','k','scattercolors','same','outliers',0,...
    'boxspacing',1,'boxwidth',1.5,'violinalpha',0.5);
ylabel('Percentage');
labels1 = [{'Winner'},{'Loser'},{'Draw'}];
labels2 = [{'wins'},{'wins'},{''}];
for i = 1:3
    text(i,16.8,labels1{i},'HorizontalAlignment','center','FontSize',font_size);
    text(i,15.05,labels2{i},'HorizontalAlignment','center','FontSize',font_size);
end
xl = xlim; 
xlim([xl(1)+0.005, xl(2)+0.05]); % make more space for the legend
ylim([18,48]);
ax3.FontSize = font_size;

%%% D) RESPONSE PLAYED %%%
ax1 = axes('Position',[0.565,0.41,0.41,0.43]);

% Make the colourmap
p_col = hot(10);
p_col = p_col([2,8,5],:);
p1 = summer(10);
p_col_rps = p1([2,6,9],:);

% Plot
plot([0,4],[(1/3),(1/3)]*100,'k','LineWidth',1,'LineStyle','--');
hold on;
h = daviolinplot([ranked_resp(1,:),ranked_resp(2,:),ranked_resp(3,:)]','groups',sort(repmat(1:3,[1,size(ranked_resp,2)]))',...
    'xtlabels', {''},'color',p_col,'scatter',2,'jitter',1,...
    'box',3,'boxcolors','k','outliers',0,'scattercolors','same',...
    'boxspacing',1,'boxwidth',1.5,'violinalpha',0.5);
ylabel('Percentage');
ylim([18,48]);
ax1.FontSize = font_size;
ax1.XAxis.TickLabelRotation = 0;
ax1.YTick = 20:5:45;
 
% Labels
labels1 = [{'Most'},{'Mid'},{'Least'}];
labels2 = [{'played'},{'played'},{'played'}];
for i = 1:3
    text(i,16.8,labels1{i},'HorizontalAlignment','center','FontSize',font_size);
    text(i,15.2,labels2{i},'HorizontalAlignment','center','FontSize',font_size);
end

% Legend for pie charts
h = gobjects(1,3);
for rps = 1:3
    h(1,rps) = fill([100,101,101,100],[100,100,101,101],p_col_rps(rps,:),'FaceAlpha',0.5);
end
legend(h,[{'Rock'},{'Paper'},{'Scissors'}]);
legend boxoff;

% Pie charts
r1 = tabulate(all_played_rank(1,:));
r2 = tabulate(all_played_rank(2,:));
r3 = tabulate(all_played_rank(3,:));
rank_played = [r1(:,3),r2(:,3),r3(:,3)];

x_pos_pie = linspace(0.583,0.872,3);
x_pos_pie = x_pos_pie(1:3);
for rps = 1:3
    ax_pie = axes('Position',[x_pos_pie(rps),0.845,0.105,0.105]);
    ph = pie(rank_played(:,rps),[{''},{''},{''}]);
    t = (rank_played(:,rps)/100)*(2*pi);
    t = cumsum(t);
    t = [[0;t(1:end-1)],t];
    t = mean(t,2)+pi*0.5;
    labels = [{'R'},{'P'},{'S'}];
    [x,y] = pol2cart(t,0.6);
    for i = 1:size(t,1)
        h = ph(i*2-1);
        h.FaceColor = p_col_rps(i,:)';
        h.FaceAlpha = 0.5;
        h.EdgeColor = 'k';
        h.LineWidth = 0.5;
        text(x(i),y(i),labels(i),'HorizontalAlignment','center','FontSize',font_size-2);
    end
    xlim([-1,1])
    ylim([-1,1])
end

%%% E) CHANGE RESPONSE SPLIT BY PREVIOUS OUTCOME %%%
prop_change = 100-prop_stay;
ax3 = axes('Position',[0.066,0.07,0.41,0.385]);
hold on

% Colourmap
p_col = parula(20);
p_col = p_col([8,5,2],:);

% Plot
plot([0,4],[(2/3),(2/3)]*100,'k','LineWidth',1,'LineStyle','--');
hold on;
h = daviolinplot_limit([prop_change(1,:),prop_change(2,:),prop_change(3,:)]','groups',sort(repmat(1:3,[1,size(prop_change,2)]))',...
    'xtlabels', {''},'color',p_col,'scatter',2,'jitter',1,...
    'box',3,'boxcolors','k','scattercolors','same','outliers',0,...
    'boxspacing',1,'boxwidth',1.5,'violinalpha',0.5,'y_limit',[nan,100]);
ylim([20,103]);
labels1 = [{'After'},{'After'},{'After'}];
labels2 = [{'win'},{'loss'},{'draw'}];
for i = 1:3
    text(i,16.8,labels1{i},'HorizontalAlignment','center','FontSize',font_size);
    text(i,11.9,labels2{i},'HorizontalAlignment','center','FontSize',font_size);
end
text(0.17,mean(ylim),'Percentage','FontSize',font_size+2,'HorizontalAlignment','center','Rotation',90);
ax3.FontSize = font_size;

%%% F) PREDICTABILITY %%%
% Make the colourmap
p_col = lines(7);
p_col = p_col(1,:);

% Plot
ax4 = axes('Position',[0.565,0.07,0.41,0.23]);
hold on
line([5,100],[1/3,1/3]*100,'Color','k','LineWidth',1,'LineStyle','--');
for ppt = 1:num_pairs*2
    plot(5:100,pred_acc(ppt,:),'Color',[0.3,0.3,0.3,0.15],'LineWidth',1);
end
fill([5:100,100:-1:5],[mean(pred_acc)+CI_pred_acc,fliplr(mean(pred_acc)-CI_pred_acc)],p_col,'FaceAlpha',0.2,'LineStyle','none');
plot(5:100,mean(pred_acc),'Color',p_col,'LineWidth',3);
ylim([25,65]);
xlim([5,100]);
xlabel('N previous games');
ylabel('Accuracy (%)');
ax4.FontSize = font_size;

%% Save the plot
fn = fullfile(path_to_data,'derivatives','plots','Figure1');
tn = tempname;
print(gcf,'-dpng','-r1000',tn)
im=imread([tn '.png']);
[ii,jj]=find(mean(im,3)<255);margin=0;
imwrite(im(min(ii-margin):max(ii+margin),min(jj-margin):max(jj+margin),:),[fn '.png'],'png');
