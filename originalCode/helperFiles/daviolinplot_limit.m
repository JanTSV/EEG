function h = daviolinplot_limit(data, varargin)
% DAVIOLINPLOT_LIMIT  Wrapper um daviolinplot, der optional y-Limits setzt.
% Akzeptiert beliebige Name-Value-Paare (z.B. 'xtlabels') und reicht sie durch.
%
% Optional:
%   'ylim', [ymin ymax]   -> setzt ylim nach dem Plot

    yLimits = [];

    % --- 'ylim' aus varargin herausziehen (wenn vorhanden) ---
    idx = find(strcmpi(varargin, 'ylim'), 1, 'first');
    if ~isempty(idx)
        if numel(varargin) < idx+1
            error('daviolinplot_limit:InvalidInput', '''ylim'' needs a [min max] value.');
        end
        yLimits = varargin{idx+1};
        varargin(idx:idx+1) = []; % entferne 'ylim' + Wert aus Argumentliste
    end

    % --- Violinplot aufrufen (alles andere unverändert durchreichen) ---
    h = daviolinplot(data, varargin{:});

    % --- y-Limits setzen, falls angegeben ---
    if ~isempty(yLimits)
        ylim(yLimits);
    end
end
