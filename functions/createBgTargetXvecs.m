function [xVecLiq, xVecBg, xVecTarget] = createBgTargetXvecs(Geometry, params)
% createBgTargetXvecs - creates background and target xVec = [muA; muS] for
% specified geometry (=mesh)
%
% Arguments:
%   Geometry    - see createGeometry function
%   params      - parameters to generate xVec (will be added later)
%
% Output:
%   xVecLiq     - xVec for phantom consisting only of liquid part (used as
%                 x0Vec for reconstructions
%   xVecBg      - xVec for background phantom (no inclusions)
%   xVecTarget  - xVec for target phantom (with inclusions)
% 
% konstantin.tamarov@uef.fi

arguments
    Geometry struct;
    % liquid phantom part absorption and scattering
    params.muALiq double = 0.0065; % mm-1
    params.muSLiq double = 0.95; % mm-1
    % solid phantom part parameters
    params.muASol double = 0.01; % mm-1
    params.muSSol double = 0.8; % mm-1
    params.hSolid double = 10; % mm
    % inclusion parameters: cylindrical inclusion with inner and outer
    % radii and height
    params.rInclusionOuter = [6 6]; % mm
    params.rInclusionInner = [6 6]; % mm
    params.zInclusion = [85, 85]; % mm
    params.xInclusion = [20 * cosd(5), 20 * cosd(-115)]; % mm
    params.yInclusion = [20 * sind(5), 20 * sind(-115)]; % mm
    params.muAInclusion = [4 * 0.0065, 5 * 0.0065]; % mm-1
    params.muSInclusion = [0.6 * 0.95, 0.5 * 0.95]; % mm-1
    params.muABoundary = [4 * 0.0065, 5 * 0.0065]; % mm-1
    params.muSBoundary = [0.6 * 0.95, 0.5 * 0.95]; % mm-1
end

[vtx, ~, ~] = Geometry.hMesh.Data();
nodeCount = Geometry.hMesh.NodeCount();
meshDim = size(vtx, 2);

% liquid part of the phantom
muAVec = params.muALiq * ones(nodeCount, 1); % mm-1
muSVec = params.muSLiq  * ones(nodeCount, 1); % mm-1
xVecLiq = [muAVec; muSVec];

% solid outer epoxy-TiO2-ink part of phantom
vtxInd = find(sqrt(vtx(:, 1) .^ 2 + vtx(:, 2) .^ 2) >= Geometry.sizes(1)/2 - params.hSolid);
muAVec(vtxInd) = 0.01; % mm-1
muSVec(vtxInd) = 0.8; % mm-1
xVecBg = [muAVec; muSVec];

% Target simulated data
for inclInd = 1:length(params.muAInclusion)
    if meshDim == 3
        % set absorption and scattering inside of inclusions
        vtxInd = find(sqrt((params.xInclusion(inclInd) - vtx(:,1 )).^2 ...
            + (params.yInclusion(inclInd) - vtx(:, 2)).^2) < params.rInclusionOuter(inclInd) ...
            & vtx(:, 3) < params.zInclusion(inclInd) / 2 & vtx(:, 3) > -params.zInclusion(inclInd) / 2);
        muAVec(vtxInd) = params.muAInclusion(inclInd);
        muSVec(vtxInd) = params.muSInclusion(inclInd);
        % set absorption and scattering on inclusion boundaries
        vtxInd = find(sqrt((params.xInclusion(inclInd) - vtx(:,1 )).^2 ...
            + (params.yInclusion(inclInd) - vtx(:, 2)).^2) <= params.rInclusionOuter(inclInd) ...
            & sqrt((params.xInclusion(inclInd) - vtx(:,1 )).^2 ...
            + (params.yInclusion(inclInd) - vtx(:, 2)).^2) >= params.rInclusionInner(inclInd) ...
            & vtx(:, 3) < params.zInclusion(inclInd) / 2 & vtx(:, 3) > -params.zInclusion(inclInd) / 2);
    end
    if meshDim == 2
        % set absorption and scattering inside of inclusions
        vtxInd = find(sqrt((params.xInclusion(inclInd) - vtx(:,1 )).^2 ...
            + (params.yInclusion(inclInd) - vtx(:, 2)).^2) < params.rInclusionOuter(inclInd));
        muAVec(vtxInd) = params.muAInclusion(inclInd);
        muSVec(vtxInd) = params.muSInclusion(inclInd);
        % set absorption and scattering on inclusion boundaries
        vtxInd = find(sqrt((params.xInclusion(inclInd) - vtx(:,1 )).^2 ...
            + (params.yInclusion(inclInd) - vtx(:, 2)).^2) <= params.rInclusionOuter(inclInd) ...
            & sqrt((params.xInclusion(inclInd) - vtx(:,1 )).^2 ...
            + (params.yInclusion(inclInd) - vtx(:, 2)).^2) >= params.rInclusionInner(inclInd));
    end

    muAVec(vtxInd) = params.muABoundary(inclInd);
    muSVec(vtxInd) = params.muSBoundary(inclInd);

end
clear vtx;
xVecTarget = [muAVec; muSVec];

end

