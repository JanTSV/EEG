%% Plot decoding
%
% Toolboxes needed: fieldtrip (we used version 20240110 here), cosmomvpa,
% bayesfactor wrapper (included, see https://github.com/LinaTeichmann1/BFF_repo).
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
test_idx = 1:4;
num_tests = size(test_idx,2);
num_time_bins = 20;

%% Load the data
all_decoding_accuracy = zeros(num_pairs*2,num_time_bins,num_tests); % Decoding for all players
all_searchlight_accuracy = cell(num_tests,num_pairs*2); % Searchlight for all players
decoding_accuracy_wl = zeros(num_pairs,num_time_bins,2,num_tests);  % Split by winners/losers
for p = 1:num_pairs
    % Get the pair ID
    pair = pair_ids(1,p);

    % Loop over the 2 players in the pair
    for ppt = 1:2
        % Load in the decoding accuracies
        load(sprintf('%s/derivatives/pair-%02d_player-%01d_task-RPS_decoding.mat',path_to_data,pair,ppt));
        % Loop over tests
        for test = 1:num_tests
            all_decoding_accuracy(pair_idx(p,ppt),:,test) = decoding_accuracy{test}.samples;
            all_searchlight_accuracy{test,pair_idx(p,ppt)} = searchlight_acc{test};
        end
    end

    % Load in the behaviour
    events = readtable(fullfile(path_to_data,num2str(pair,'sub-%02d'),'eeg',num2str(pair,'sub-%02d_task-RPS_events.tsv')),'Filetype','text','delimiter','\t');
    winner_idx = find([sum(events.outcome==2)>sum(events.outcome==3),sum(events.outcome==2)<sum(events.outcome==3),sum(events.outcome==2)==sum(events.outcome==3)]);
    if winner_idx==3
        warning('There is no winner for this pair');
    end
  
    % Split the decoding by whether the player won or lost the game
    this_pair_idx = pair_idx(p,:);    
    for test = 1:num_tests
        % Winner
        decoding_accuracy_wl(p,:,1,test) = all_decoding_accuracy(this_pair_idx(winner_idx),:,test);
        % Loser
        decoding_accuracy_wl(p,:,2,test) = all_decoding_accuracy(this_pair_idx(~ismember(1:2,winner_idx)),:,test);
    end
end % Loop over pairs

% Stack the searchlight results
searchlight_accuracy = cell(4,1);
for test = 1:num_tests
    searchlight_accuracy{test,1} = cosmo_stack(all_searchlight_accuracy(test,:),1);
end

%% Calculate the Bayes Factors
% See https://github.com/LinaTeichmann1/BFF_repo
bf = zeros(num_tests,num_time_bins);
bf_wl = zeros(3,num_time_bins,num_tests);
for test = 1:num_tests
    % All ppts
    bf(test,:) = bayesfactor_R_wrapper((all_decoding_accuracy(:,:,test)-(1/3))','args','mu=0,rscale="medium",nullInterval=c(0.5,Inf)');

    % Winners vs. losers
    for win_lose = 1:2
        bf_wl(win_lose,:,test) = bayesfactor_R_wrapper((decoding_accuracy_wl(:,:,win_lose,test)-(1/3))','args','mu=0,rscale="medium",nullInterval=c(0.5,Inf)');
    end
    % Difference between winners & losers
    bf_wl(3,:,test) = bayesfactor_R_wrapper((decoding_accuracy_wl(:,:,1,test)-decoding_accuracy_wl(:,:,2,test))','returnindex',2,'args','mu=0,rscale="medium",nullInterval=c(-0.5,0.5)');
end

%% Figure 2: Plot decoding
% Set the figure positions 
fh = figure(2);clf;
fh.Position = [100,100,800,800];
test_idx = 1:4;
x_pos_plot = [0.07,0.535,0.07,0.535];
y_pos_plot = [0.61,0.61,0.11,0.11]+0.056;
x_pos_sl = linspace(0,0.335,6);
x_pos_sl = [x_pos_sl(1:2)+0.0145,x_pos_sl(3:4)+0.0255,x_pos_sl(5)+0.0335];
aw = 0.335./5;
bfh = .1;

% Time parameters
%tv = (0.25:0.25:5)-(0.25/2);
tv_idx = [{1:8},{9:16},{17:20}];
timebin_sl =  [1:4:17;4:4:20]';

% Labels
font_size = 14;
font_size_legend = 10;
time_labels = [{'Decision'},{'Response'},{'Feedback'}];
titles = [{'A) Own response'},{'B) Opponent''s response'},{'C) Own previous response'},{'D) Opponent''s previous response'}];

% Make the colourmap
% We used these colourmaps: https://www.mathworks.com/matlabcentral/fileexchange/120088-200-colormap
% Switching to standard colourmaps here.
pcol = lines(7);
pcol = pcol([3,2,4],:);
p_col_bf = [[linspace(0.03,1,50);linspace(0.30,1,50);linspace(0.6,1,50)],[linspace(1,0.62,50);linspace(1,0.05,50);linspace(1,0.08,50)]]'; % There's no good standard colourmap for this, so we make one from scratch (red to blue via white)
cm_searchlight = hot(110);
cm_searchlight = cm_searchlight(1:100,:); % Remove white to make it more similar to the one we used
bf_lim = 8; % BF limits (on a log scale)
val_col_map = logspace(-bf_lim,bf_lim,size(p_col_bf,1));

% Loop over the different things we decode
for test = 1:size(test_idx,2)

    % Title
    ax_t = axes('Position',[x_pos_plot(test)-0.065,y_pos_plot(test)+0.315,0.38,0.25],'LineWidth',1.5);
    text(0,0,titles{test},'HorizontalAlignment','left','FontSize',font_size+2,'FontWeight','bold');
    ax_t.Visible = 'off';

    % Make the axes
    ax = axes('Position',[x_pos_plot(test),y_pos_plot(test),0.38,0.29],'LineWidth',1.5);
    hold on;
    plot([0,num_time_bins+1],[1/3,1/3]*100,'k','LineWidth',1.5,'LineStyle','--');

    % We want different colours for the 3 different phases
    for tb = 1:3
        % Get the decoding accuracy for this test & time bin
        res_sel = all_decoding_accuracy(:,tv_idx{tb},test)*100;
        % Background
        fill([tv_idx{tb}(1),tv_idx{tb}(end),tv_idx{tb}(end),tv_idx{tb}(1)]+[-0.25,0.25,0.25,-0.25],[30,30,40,40],[0.5,0.5,0.5],'FaceAlpha',0.1,'LineStyle','none');
        text(mean(tv_idx{tb}),39.7,time_labels{tb},'FontSize',font_size-2,'HorizontalAlignment','center','FontWeight','bold');
        % Plot the data
        CI = (std(res_sel)/sqrt(size(res_sel,1))).*repmat(tinv(0.975,size(res_sel,1)-1),[size(res_sel,2),1])';
        fill([tv_idx{tb},fliplr(tv_idx{tb})],[mean(res_sel)-CI,fliplr(mean(res_sel)+CI)],pcol(tb,:),'FaceAlpha',0.15,'LineStyle','none');
        plot(tv_idx{tb},mean(res_sel),'color',pcol(tb,:),'LineWidth',1.5);
        scatter(tv_idx{tb},mean(res_sel),'MarkerEdgeColor',pcol(tb,:),'MarkerFaceColor',pcol(tb,:),'MarkerFaceAlpha',0.5,'LineWidth',1,'SizeData',46);
    end

    % Set the axes parameters
    xlim([0,21]);
    ylim([31,40]);
    ax.XTick = 0.5:4:20.5;
    ax.XTickLabel = [];
    ax.FontSize = font_size;
    text(-2.8,mean(ylim),'Decoding accuracy (%)','FontSize',font_size,'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','bottom');

    %%% BFs %%%
    ax_bf = axes('Position',[x_pos_plot(test),y_pos_plot(test)-0.038,0.38,0.025],'LineWidth',1.5);
    hold on

    % Plot the grey background showing the 3 phases
    for tb = 1:3
        fill([tv_idx{tb}(1),tv_idx{tb}(end),tv_idx{tb}(end),tv_idx{tb}(1)]+[-0.25,0.25,0.25,-0.25],[-bf_lim,-bf_lim,bf_lim,bf_lim],[0.5,0.5,0.5],'FaceAlpha',0.1,'LineStyle','none');
    end

    % Plot the BFs
    plot(0:num_time_bins+1,zeros(1,num_time_bins+2),'Color','k','LineWidth',1.5);
    for i = 1:num_time_bins
        [~,idx] = min(abs(val_col_map-bf(test,i)));
        scatter(i,log10(bf(test,i)),[],[0.5,0.5,0.5],'filled','SizeData',100);
        scatter(i,log10(bf(test,i)),[],p_col_bf(idx,:),'filled','SizeData',80);
    end

    % Set the axes parameters
    xlim([0,21]);
    ylim([-bf_lim,bf_lim]);
    ax_bf.XTick = 0.5:4:20.5;
    ax_bf.XTickLabel = [];
    ax_bf.XTickLabelRotation = 0;
    ax_bf.YTick = [-bf_lim,bf_lim];
    ax_bf.YTickLabel = [{['10^{',num2str(-bf_lim),'}']},{['10^{',num2str(bf_lim),'}']}];
    ax_bf.FontSize = font_size;
    ax_bf.Clipping = 'off';
    text(-2.8,0,'BF','FontSize',font_size,'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','bottom');
    text(-2.25,0,'(log scale)','FontSize',font_size_legend,'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','bottom');

    % Add a colour bar (but only for the plots on the right)
    if mod(test,2)==0
        % Labels
        text(22.2,3.7,'BF (log','FontSize',font_size_legend,'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','bottom');
        text(22.75,3.7,'scale)','FontSize',font_size_legend,'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','bottom');
        % Colourbar
        colormap(ax_bf,p_col_bf);
        cbh = colorbar;
        cbh.Position = [ax_bf.Position(1)+ax_bf.Position(3)+0.033,y_pos_plot(test)-0.0385,0.01,0.0395];
        cbh.Ticks = [0.1,0.9];
        cbh.TickLength = 0;
        cbh.TickLabels = [{['10^{',num2str(-bf_lim),'}']},{['10^{',num2str(bf_lim),'}']}];
        cbh.FontSize = font_size_legend;
    end

    %%% SEARCHLIGHT %%%
    a_sl = axes('Position',[x_pos_plot(test),y_pos_plot(test)-0.115,0.38,0.7*bfh],'LineWidth',1.5);
    hold on;
    xlim([0,21]);
    ylim([0,1]);
    a_sl.XTick = 0.5:4:20.5;
    a_sl.XTickLabel = 0:5;
    text(10.5,-0.5,'Time (s)','HorizontalAlignment','center','FontSize',font_size);
    a_sl.FontSize = font_size;
    a_sl.YAxis.Visible = 'off';
    % Plot the grey background showing the 3 phases
    for tb = 1:3
        fill([tv_idx{tb}(1),tv_idx{tb}(end),tv_idx{tb}(end),tv_idx{tb}(1)]+[-0.25,0.25,0.25,-0.25],[-bf_lim,-bf_lim,bf_lim,bf_lim],[0.5,0.5,0.5],'FaceAlpha',0.1,'LineStyle','none');
    end
    if mod(test,2)==0
        text(22.2,0.47,'Decoding','FontSize',font_size_legend,'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','bottom');
        text(22.75,0.47,'acc. (%)','FontSize',font_size_legend,'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','bottom');
    end

    % Get the data for this test
    res = searchlight_accuracy{test,1};
    ft = ft_timelockanalysis([],cosmo_map2meeg(res));
    ft.time = 1:20;

    % Find the electrode layout
    layout = cosmo_meeg_find_layout(res,'label_threshold',.99);
    % Select the electrodes we use
    idx = ismember(layout.label,res.a.fdim.values{1});
    layout.pos = layout.pos(idx,:);
    layout.width = layout.width(idx,:);
    layout.height = layout.height(idx,:);
    layout.label = layout.label(idx,:);

    % Loop over time bins
    for ttt = 1:size(timebin_sl,1)

        % Set the position for this figure
        a = axes('Position',[x_pos_sl(ttt)+x_pos_plot(test),y_pos_plot(test)-0.13,aw,0.98*bfh]);hold on
        a.Clipping = 'off';

        % Make the topo plot
        cfg = [];
        cfg.zlim = [1/3,0.36];
        cfg.xlim = [timebin_sl(ttt,1),timebin_sl(ttt,2)];
        cfg.layout = layout;
        cfg.showscale = 'no';
        cfg.comment = 'no';
        cfg.markersymbol = '.';
        cfg.figure = 'gca';
        cfg.style = 'straight';
        cfg.gridscale = 128;
        ft_topoplotER(cfg, ft);
        a.FontSize = 12;
        set(a.Children,'LineWidth',.5)
        colormap(gca,cm_searchlight);
    end

    % Add the colourbar (for the plots on the right)
    if mod(test,2)==0
        colormap(a,cm_searchlight);
        cbh = colorbar;
        cbh.Position = [ax_bf.Position(1)+ax_bf.Position(3)+0.033,y_pos_plot(test)-0.1155,0.01,bfh*0.68];
        cbh.Ticks = [cfg.zlim(1)+(diff(cfg.zlim)/10),cfg.zlim(2)-(diff(cfg.zlim)/10)];
        cbh.TickLength = 0;
        cbh.TickLabels = [{sprintf('%.1f',cfg.zlim(1)*100)},{sprintf('%.1f',cfg.zlim(2)*100)}];
        cbh.FontSize = font_size_legend;
    end

end

% Save the plot
fn = fullfile(path_to_data,'derivatives','plots','Figure2');
tn = tempname;
print(gcf,'-dpng','-r1000',tn)
im=imread([tn '.png']);
[ii,jj]=find(mean(im,3)<255);margin=0;
imwrite(im(min(ii-margin):max(ii+margin),min(jj-margin):max(jj+margin),:),[fn '.png'],'png');

%% Figure 3: Plot decoding, split by winners and losers
% Set the figure positions 
fh = figure(3);clf;
fh.Position = [100,100,800,800];
test_idx = 1:4;
x_pos_plot = [0.07,0.535,0.07,0.535];
y_pos_plot = [0.61,0.61,0.11,0.11]+0.056;
y_pos_bf = [y_pos_plot-0.038;y_pos_plot-0.0765;y_pos_plot-0.115];

% Time parameters
tv = (0.25:0.25:5)-(0.25/2);
tv_idx = [{1:8},{9:16},{17:20}];

% Labels
font_size = 14;
font_size_legend = 10;
time_labels = [{'Decision'},{'Response'},{'Feedback'}];
titles = [{'A) Own response'},{'B) Opponent''s response'},{'C) Own previous response'},{'D) Opponent''s previous response'}];
bf_labels = [{'Winners'},{'Losers'},{'Difference'}];

% Make the colourmap
% We used these colourmaps in the paper: https://www.mathworks.com/matlabcentral/fileexchange/120088-200-colormap
% Switching to standard colourmaps here.
pcol = winter(10);
pcol = pcol([3,8],:);
p_col_bf = [[linspace(0.03,1,50);linspace(0.30,1,50);linspace(0.6,1,50)],[linspace(1,0.62,50);linspace(1,0.05,50);linspace(1,0.08,50)]]'; % There's no good standard colourmap for this, so we make one from scratch (red to blue via white)
bf_lim = 6; % BF limits (on a log scale)
val_col_map = logspace(-bf_lim,bf_lim,size(p_col_bf,1));

% Loop over the different things we decode
for test = 1:size(test_idx,2)

    % Title
    ax_t = axes('Position',[x_pos_plot(test)-0.065,y_pos_plot(test)+0.315,0.38,0.25],'LineWidth',1.5);
    text(0,0,titles{test},'HorizontalAlignment','left','FontSize',font_size+2,'FontWeight','bold');
    ax_t.Visible = 'off';

    % Make the axes
    ax = axes('Position',[x_pos_plot(test),y_pos_plot(test),0.38,0.29],'LineWidth',1.5);
    hold on;
    plot([0,num_time_bins+1],[1/3,1/3]*100,'k','LineWidth',1.5,'LineStyle','--');

    % Plot the grey backgrounds for the 3 different phases
    for tb = 1:3
        fill([tv_idx{tb}(1),tv_idx{tb}(end),tv_idx{tb}(end),tv_idx{tb}(1)]+[-0.25,0.25,0.25,-0.25],[30,30,40,40],[0.5,0.5,0.5],'FaceAlpha',0.1,'LineStyle','none');
        text(mean(tv_idx{tb}),39.7,time_labels{tb},'FontSize',font_size-2,'HorizontalAlignment','center','FontWeight','bold');
    end

    % Loop over winners/losers
    ph = gobjects(1,2);
    for win_lose = 1:2
        for tb = 1:3
            % Get the decoding accuracy for this test & time bin
            res_sel = decoding_accuracy_wl(:,tv_idx{tb},win_lose,test)*100;
            % Plot the data
            CI = (std(res_sel)/sqrt(size(res_sel,1))).*repmat(tinv(0.975,size(res_sel,1)-1),[size(res_sel,2),1])';
            fill([tv_idx{tb},fliplr(tv_idx{tb})],[mean(res_sel)-CI,fliplr(mean(res_sel)+CI)],pcol(win_lose,:),'FaceAlpha',0.15,'LineStyle','none');
            ph(1,win_lose) = plot(tv_idx{tb},mean(res_sel),'color',pcol(win_lose,:),'LineWidth',1.5);
            scatter(tv_idx{tb},mean(res_sel),'MarkerEdgeColor',pcol(win_lose,:),'MarkerFaceColor',pcol(win_lose,:),'MarkerFaceAlpha',0.5,'LineWidth',1,'SizeData',46);
        end
    end

    % Set the axes parameters
    xlim([0,21]);
    ylim([31,40]);
    ax.XTick = 0.5:4:20.5;
    ax.XTickLabel = [];
    ax.FontSize = font_size;
    text(-2.8,mean(ylim),'Decoding accuracy (%)','FontSize',font_size,'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','bottom');
    lh = legend(ph,[{'Winners'},{'Losers'}],'Location','SE');
    if test>2
        lh.Position(2) = lh.Position(2)+0.2; % Shift legend to a better position
    end
    legend boxoff;

    %%% BFs %%%
    for bf_idx = 1:3 % W - L - Diff
        ax_bf = axes('Position',[x_pos_plot(test),y_pos_bf(bf_idx,test),0.38,0.025],'LineWidth',1.5);
        hold on
        % Plot the grey background showing the 3 phases
        for tb = 1:3
            fill([tv_idx{tb}(1),tv_idx{tb}(end),tv_idx{tb}(end),tv_idx{tb}(1)]+[-0.25,0.25,0.25,-0.25],[-bf_lim,-bf_lim,bf_lim,bf_lim],[0.5,0.5,0.5],'FaceAlpha',0.1,'LineStyle','none');
        end
        % Plot the BFs
        plot(0:num_time_bins+1,zeros(1,num_time_bins+2),'Color','k','LineWidth',1.5);
        for i = 1:num_time_bins
            [~,idx] = min(abs(val_col_map-bf_wl(bf_idx,i,test)));
            scatter(i,log10(bf_wl(bf_idx,i,test)),[],[0.5,0.5,0.5],'filled','SizeData',100);
            scatter(i,log10(bf_wl(bf_idx,i,test)),[],p_col_bf(idx,:),'filled','SizeData',80);
        end

        % Set the axes parameters
        xlim([0,21]);
        ylim([-bf_lim,bf_lim]);
        ax_bf.XTick = 0.5:4:20.5;
        ax_bf.XTickLabel = [];
        ax_bf.XTickLabelRotation = 0;
        ax_bf.YTick = [-bf_lim,bf_lim];
        ax_bf.YTickLabel = [{['10^{',num2str(-bf_lim),'}']},{['10^{',num2str(bf_lim),'}']}];
        ax_bf.FontSize = font_size;
        ax_bf.YAxis.FontSize = font_size_legend;
        ax_bf.Clipping = 'off';
        text(21,bf_lim+2,bf_labels{bf_idx},'HorizontalAlignment','right','FontSize',font_size_legend+1,'FontWeight','bold');

        % Add colour bar (to plots on the right, only 1 for all 3 BF plots)
        if bf_idx==2
            text(-2.8,0,'BF','FontSize',font_size,'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','bottom');
            text(-2.25,0,'(log scale)','FontSize',font_size_legend,'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','bottom');
            if mod(test,2)==0
                text(22.2,0,'BF','FontSize',font_size,'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','bottom');
                text(22.75,0,'(log scale)','FontSize',font_size_legend,'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','bottom');
                colormap(ax_bf,p_col_bf);
                cbh = colorbar;
                cbh.Position = [ax_bf.Position(1)+ax_bf.Position(3)+0.033,y_pos_bf(3,test),0.01,0.103];
                cbh.Ticks = [0,0.5,1];
                cbh.TickLength = 0;
                cbh.TickLabels = [{['10^{',num2str(-bf_lim),'}']},{'1'},{['10^{',num2str(bf_lim),'}']}];
                cbh.FontSize = font_size_legend;
            end    
        elseif bf_idx==3
            ax_bf.XTick = 0.5:4:20.5;
            ax_bf.XTickLabel = 0:5;
            text(10.5,-22,'Time (s)','HorizontalAlignment','center','FontSize',font_size);

        end
    end
end

% Save the plot
fn = fullfile(path_to_data,'derivatives','plots','Figure3');
tn = tempname;
print(gcf,'-dpng','-r1000',tn)
im=imread([tn '.png']);
[ii,jj]=find(mean(im,3)<255);margin=0;
imwrite(im(min(ii-margin):max(ii+margin),min(jj-margin):max(jj+margin),:),[fn '.png'],'png');
