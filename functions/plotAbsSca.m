function [fig, image] = plotAbsSca(xVec, Geometry, params)
% plotAbsSca - master function to plot 2D or 3D xVec = [muA; muS] data
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
% Output:
%   
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

if length(Geometry.sizes) > 2
    fig = plot2DSlices(xVec, Geometry, params);
    return;
end

if params.analysisType == AnalysisType.Absorption
    numParams = 1;
    muaRecon = xVec;
    image = zeros([Geometry.dims' 1]);
else
    numParams = 2;
    muaRecon = xVec(1:end/2);
    musRecon = xVec(end/2+1:end);
    if params.isGridBasis == 0
        musRecon = reshape(Geometry.hBasis.Map('M->B', musRecon), Geometry.dims(1), Geometry.dims(2));
    else
        musRecon = zeros(Geometry.dims(1) * Geometry.dims(2), 1);
        musRecon(Geometry.hBasis.GridElref() > 0) = xVec(end/2+1:end);
        musRecon = reshape(musRecon, Geometry.dims(1), Geometry.dims(2));
    end
    minScaColorbar = min(musRecon, [], 'all') - eps;
    maxScaColorbar = max(musRecon, [], 'all') + eps;
    image = zeros([Geometry.dims' 2]);
    image(:, :, 2) = musRecon.';
end

if params.isGridBasis == 0
    muaRecon = reshape(Geometry.hBasis.Map('M->B', muaRecon), Geometry.dims(1), Geometry.dims(2));
else
    tmp = zeros(Geometry.dims(1) * Geometry.dims(2), 1);
    tmp(Geometry.hBasis.GridElref() > 0) = muaRecon;
    muaRecon = reshape(tmp, Geometry.dims(1), Geometry.dims(2));
end
minAbsColorbar = min(muaRecon, [], 'all') - eps;
maxAbsColorbar = max(muaRecon, [], 'all') + eps;
image(:, :, 1) = muaRecon.';

if params.plotType == PlotType.Difference
    reconstructedType = '\delta';
else
    reconstructedType = '';
end

fig = figure('Position', [0, 0, numParams * 600, 500], ...
    'Units', 'pixels', 'Name', ['nFreqs ' num2str(length(Geometry.freqsVec))], ...
    'PaperSize', [8.5 8.5/300*250], 'PaperUnits', 'centimeters');
tcl = tiledlayout(1, numParams, "TileSpacing", "compact", "Padding", "compact");
if strlength(params.figTitle) > 0
    title(tcl, params.figTitle, 'FontWeight', 'bold');
end

nexttile();
absSlice = muaRecon';
img = imagesc(Geometry.gridCornerCoords{1}, ...
    Geometry.gridCornerCoords{2}, ...
    absSlice, [minAbsColorbar maxAbsColorbar]);
set(img, 'AlphaData', absSlice ~= 0);
axis square;
set(gca, "YDir", "normal");
xlabel('x [mm]'); ylabel('y [mm]');
set(gca, 'Colormap', parula, 'CLim', [minAbsColorbar maxAbsColorbar], 'FontSize', 14);
title({' ' ' '});
cb = colorbar();
cb.Label.String = {[reconstructedType '\mu_a [mm^{-1}]'], '    x10^{-3}'};
cb.Label.FontSize = 14;
cb.Label.VerticalAlignment = 'middle';
cb.Label.Position = [0.5 maxAbsColorbar+0.1*(maxAbsColorbar - minAbsColorbar)];
cb.Label.Rotation = 0;
cb.TickLabels = cb.Ticks * 1000;

if numParams > 1
    nexttile();
    absSlice = musRecon';
    img = imagesc(Geometry.gridCornerCoords{1}, ...
        Geometry.gridCornerCoords{2}, ...
        absSlice, [minScaColorbar maxScaColorbar]);
    set(img, 'AlphaData', absSlice ~= 0);
    axis square;
    set(gca, "YDir", "normal");
    xlabel('x [mm]'); ylabel('y [mm]');
    set(gca, 'Colormap', parula, 'CLim', [minScaColorbar maxScaColorbar], 'FontSize', 14);
    title({' ' ' '});
    cb = colorbar();
    cb.Label.String = [reconstructedType '\mu_s [mm^{-1}]'];
    cb.Label.FontSize = 14;
    cb.Label.VerticalAlignment = 'middle';
    cb.Label.Position = [0.5 maxScaColorbar+0.14*(maxScaColorbar-minScaColorbar)];
    cb.Label.Rotation = 0;
end

end

