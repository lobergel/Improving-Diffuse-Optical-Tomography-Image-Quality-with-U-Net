function figData = plotDifferenceYData(dataBg, dataTarget, params)
% plotMeasuredData - plot the y data from experiment or simulation
%
% Arguments:
%   dataBg      - data of phantom with no inclusions, can be cell array
%   dataTarget  - data of phantom with inclusions, can be cell array
%   params      - optional arguments (see arguments block)
%
% konstantin.tamarov@uef.fi

arguments (Repeating)
    dataBg;
    dataTarget;
end
arguments
    params.figTitle string = "";
    params.legends string = [];
    % cell arrays for the errors of data
    % should be either empty or contain the same number of vectors as
    % dataBg and dataTrget, respectively
    params.errorsBg = {};
    params.errorsTarget = {};
end

if ~iscell(params.errorsBg)
    params.errorsBg = {params.errorsBg};
end
if ~iscell(params.errorsTarget)
    params.errorsTarget = {params.errorsTarget};
end
if isempty(params.errorsBg)
    params.errorsBg = cell(size(dataBg));
    for i = 1:length(dataBg)
        params.errorsBg{i} = zeros(size(dataBg{i}));
    end
end
if isempty(params.errorsTarget)
    params.errorsTarget = cell(size(dataTarget));
    for i = 1:length(dataTarget)
        params.errorsTarget{i} = zeros(size(dataTarget{i}));
    end
end

% calculate difference data
dataDiff = cell(1, length(dataBg));
errorsDiff = cell(size(dataDiff));
for i = 1:length(dataBg)
    dataDiff{i} = dataTarget{i} - dataBg{i};
    errorsDiff{i} = sqrt(params.errorsTarget{i}.^2 + params.errorsBg{i}.^2);
end

allData = {dataBg dataTarget dataDiff};
allErrors = {params.errorsBg params.errorsTarget errorsDiff};

ylabels = ["Background" "Target" "Difference"];

% find common y limits in data
ylimAmpl = cell(1, 3); % for Bg, Target and Target-Bg
ylimPhase = cell(size(ylimAmpl));
for j = 1:3
    for i = 1:length(allData{j})
        dataMin = allData{j}{i} - allErrors{j}{i};
        dataMax = allData{j}{i} + allErrors{j}{i};
        if i == 1
            ylimAmpl{j} = [min(dataMin(1:end/2)) max(dataMax(1:end/2))];
            ylimPhase{j} = [min(dataMin(end/2+1:end)) max(dataMax(end/2+1:end))];
        else
            ylimAmpl{j} = [min([ylimAmpl{j}(1); dataMin(1:end/2)]) ...
                max([ylimAmpl{j}(2); dataMax(1:end/2)])];
            ylimPhase{j} = [min([ylimPhase{j}(1); dataMin(end/2+1:end)]) ...
                max([ylimPhase{j}(2); dataMax(end/2+1:end)])];
        end
    end
end

figData = figure('Position', [50 50 1800 900], ...
    'PaperSize', [17 12], 'PaperUnits', 'centimeters');
set(figData, "Color", "white");
outer_tl = tiledlayout(1, 2, "TileSpacing", "compact", "Padding", "compact");
if strlength(params.figTitle) > 0
    title(outer_tl, params.figTitle, "FontSize", 14, "FontWeight", "bold");
end
ampl_tl = tiledlayout(outer_tl, 3, 1, "TileSpacing", "none");
ampl_tl.Layout.Tile = 1;
title(ampl_tl, "Amplitude");
phase_tl = tiledlayout(outer_tl, 3, 1, "TileSpacing", "none");
phase_tl.Layout.Tile = 2;
title(phase_tl, "Phase");

for j = 1:3
    axAmpl = nexttile(ampl_tl);
    axPhase = nexttile(phase_tl);
    for i = 1:length(allData{j})
        errorbar(axAmpl, allData{j}{i}(1:end/2), allErrors{j}{i}(1:end/2), ...
            "LineWidth", 1.5, "CapSize", 5*(sum(allErrors{j}{i}(1:end/2)) > 0));
        hold(axAmpl, "on");
        xlim(axAmpl, [0 length(allData{j}{i})/2]); ylim(axAmpl, ylimAmpl{j});
        xlabel(axAmpl, "source-detector pair"); ylabel(axAmpl, ylabels(j) + " [ln(V)]");
        set(axAmpl, 'FontSize', 12, 'TickDir', 'in', 'LabelFontSizeMultiplier', 1.1, ...
            'LineWidth', 1, 'Box' , 'on', 'FontName', 'Arial');
        

        errorbar(axPhase, allData{j}{i}(end/2+1:end), allErrors{j}{i}(end/2+1:end), ...
            "LineWidth", 1.5, "CapSize", 5*(sum(allErrors{j}{i}(end/2+1:end)) > 0));
        hold(axPhase, "on");
        xlim(axPhase, [0 length(allData{j}{i})/2]); ylim(axPhase, ylimPhase{j});
        xlabel(axPhase, "source-detector pair"); ylabel(axPhase, ylabels(j) + " [rad]");
        set(axPhase, 'FontSize', 12, 'TickDir', 'in', 'LabelFontSizeMultiplier', 1.1, ...
            'LineWidth', 1, 'Box' , 'on', 'FontName', 'Arial');
    end
    if j < 3
        axAmpl.XAxis.Label.String = ''; axAmpl.XAxis.TickValues = [];
        axPhase.XAxis.Label.String = ''; axPhase.XAxis.TickValues = [];
    end
    if j == 3
        if ~isempty(params.legends)
            legend(axAmpl, params.legends);
            legend(axPhase, params.legends);
        end
    end
end

end

