function fig = plot2DSlices(xVec, Geometry, params) % zSlices, figTitle, reconstructedParam, reconstructedType)
% plot2DReconstructions - plots 2D images of XY planes for the given array
% of Z coordinates
%
% Arguments:
%   xVec                - vector with muA or stacked muA, muS
%   Geometry            - Geometry of the problem including Toast hMesh, hBasis
%                         and sizes (xSize, ySize, zSize)
%   zSlices             - vector of Z coordinates that can vary from -zSize/2 
%                         to zSize/2 (i.e. center is at Z = 0)
%   figTitle            - title to put on top of the figure
%   analysisType        - 'Absorption' or 'AbsorptionScattering' if muA or muA and muS 
%                         were reconstructed
%   plotType            - 'Absolute' or 'Diffrence' for type of reconstruction
%   isGridBasis         - if xVec is already in grid basis
%
% konstantin.tamarov@uef.fi

arguments
    xVec double;
    Geometry struct;
    params.zSlices = [-5 0 5]; % mm, can be empty for 2D plotting
    params.figTitle string = ""; % title for figure
    params.plotType PlotType = PlotType.Difference;
    params.analysisType AnalysisType = AnalysisType.AbsorptionScattering;
    params.isGridBasis = 0;
end

% find indices in grid that are not in mesh
tmp = ones(Geometry.hMesh.NodeCount(), 1);
tmp = Geometry.hBasis.Map('M->B', tmp);
tmp = reshape(tmp, Geometry.dims(1), Geometry.dims(2), Geometry.dims(3));
idxInMesh = find(tmp > 1);

if params.analysisType == AnalysisType.Absorption
    numParams = 1;
    muaRecon = xVec;
else
    numParams = 2;
    muaRecon = xVec(1:end/2);
    musRecon = xVec(end/2+1:end);
    if params.isGridBasis == 0
        musRecon = reshape(Geometry.hBasis.Map('M->B', musRecon), Geometry.dims(1), Geometry.dims(2), Geometry.dims(3));
    end
    minScaColorbar = min(musRecon(idxInMesh), [], 'all') - eps;
    maxScaColorbar = max(musRecon(idxInMesh), [], 'all') + eps;
end
if params.isGridBasis == 0
    muaRecon = reshape(Geometry.hBasis.Map('M->B', muaRecon), Geometry.dims(1), Geometry.dims(2), Geometry.dims(3));
end
minAbsColorbar = min(muaRecon(idxInMesh), [], 'all') - eps;
maxAbsColorbar = max(muaRecon(idxInMesh), [], 'all') + eps;

if params.plotType == PlotType.Difference
    reconstructedType = '\delta';
else
    reconstructedType = '';
end

fig = figure('Position', [0, 0, numParams * 300, length(params.zSlices) * 250], ...
    'Units', 'pixels', 'Name', ['nFreqs ' num2str(length(Geometry.freqsVec))], ...
    'PaperSize', [8.5 8.5/300*250], 'PaperUnits', 'centimeters');
outer_tl = tiledlayout(1, numParams, "TileSpacing", "compact", "Padding", "compact");%, ...
    % "OuterPosition", [0 0 1 0.97]);
nSlices = length(params.zSlices);

inner_tl = tiledlayout(outer_tl, nSlices, 1, "TileSpacing", "compact", "Padding", "compact");
inner_tl.Layout.Tile = 1;
ax = gobjects(nSlices, 1);

for i = nSlices:-1:1 % opposite order so that larger z are on top of figure
    % z index for the muaRecon and musRecon
    zIndex = params.zSlices(i) + Geometry.sizes(3) / 2;
    % tile with absorption
    ax(i) = nexttile(inner_tl);
    absSlice = muaRecon(:, :, zIndex)';
    img = imagesc(Geometry.gridCornerCoords{1}, ...
        Geometry.gridCornerCoords{2}, ...
        absSlice, [minAbsColorbar maxAbsColorbar]);
    set(img, 'AlphaData', absSlice ~= 0);
    axis square;
    set(gca, "YDir", "normal");
    xlabel('x [mm]'); ylabel('y [mm]');
    % title([reconstructedType '\mu_a; z = ' num2str(zSlices(i)) ' mm']);
    title(['z = ' num2str(params.zSlices(i)) ' mm']);
    if numParams > 1
        title(inner_tl, ' '); % 'absorption');
    end
end
set(ax, 'Colormap', parula, 'CLim', [minAbsColorbar maxAbsColorbar], 'FontSize', 14);
cb = colorbar(ax(end));
cb.Layout.Tile = 'east';
cb.Label.String = {[reconstructedType '\mu_a [mm^{-1}]'], '    x10^{-3}'};
cb.Label.FontSize = 14;
cb.Label.VerticalAlignment = 'middle';
cb.Label.Position = [0.5 maxAbsColorbar+0.05*(maxAbsColorbar - minAbsColorbar)];
cb.Label.Rotation = 0;
cb.TickLabels = cb.Ticks * 1000;
% cb.Label.Rotation = 270;
% cb.Label.Position(1) = cb.Label.Position(1) + 1.05;

if numParams > 1
    % also plot scattering
    inner_tl = tiledlayout(outer_tl, nSlices, 1, "TileSpacing", "compact", "Padding", "compact");
    inner_tl.Layout.Tile = 2;
    ax = gobjects(nSlices, 1);
    for i = nSlices:-1:1 % opposite order so that larger z are on top of figure
        % z index for the muaRecon and musRecon
        zIndex = params.zSlices(i) + Geometry.sizes(3) / 2;
        
        % tile with absorption
        ax(i) = nexttile(inner_tl);
        scaSlice = musRecon(:, :, zIndex)';
        img = imagesc(Geometry.gridCornerCoords{1}, ...
            Geometry.gridCornerCoords{2}, ...
            scaSlice, [minScaColorbar maxScaColorbar]);
        set(img, 'AlphaData', scaSlice ~= 0);
        axis square;
        set(gca, "YDir", "normal");
        xlabel('x [mm]'); ylabel('y [mm]');
        % title([reconstructedType '\mu_a; z = ' num2str(zSlices(i)) ' mm']);
        title(['z = ' num2str(params.zSlices(i)) ' mm']);
    end
    set(ax, 'Colormap', parula, 'CLim', [minScaColorbar maxScaColorbar], 'FontSize', 14);
    cb = colorbar(ax(end)); 
    cb.Layout.Tile = 'east';
    cb.Label.String = [reconstructedType '\mu_s [mm^{-1}]'];
    cb.Label.FontSize = 14;
    cb.Label.VerticalAlignment = 'middle';
    cb.Label.Position = [0.5 maxScaColorbar+0.07*(maxScaColorbar-minScaColorbar)];
    cb.Label.Rotation = 0;
    % cb.Label.Rotation = 270;
    % cb.Label.Position(1) = cb.Label.Position(1) + 1.05;
    title(inner_tl, ' '); % scattering');
end
title(outer_tl, textwrap(string(params.figTitle), round(numParams * 300 / 10)), ...
    'FontWeight', 'bold');

% t = title(outer_tl, 'blah blah');
% t.Position(1) = t.Position(1) - 5;

end

