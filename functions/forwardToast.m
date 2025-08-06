function [dataSim, fluenceSim] = forwardToast(Geometry, xVec, params)
% forwardToast - sumilate forward data with Toast++, works only in mesh,
% but gird mesh can be used as reconstruction basis
%
% Arguments:
%   Geometry        - structure containing the geometry of the model
%                     including toastMesh, toastBasis to map mesh to 
%                     regular pixel grid for forward model, qVec matrix 
%                     of sources, mVec matrix of detectors, refVec vector
%                     of refractive intices for each node in mesh, freqsVec
%                     vector of frequencies to be used for reconstruction
%   xVec            - stacked [muAVec; muSVec] 
%   params.stdError - standard deviation for the simulated data
%
% Output:
%   dataSim         - simulated measured data on detectors
%   fluenceSim      - simulated fluence in the domain
%
% konstantin.tamarov@uef.fi

arguments
    Geometry struct; % data structure describing geometry
    xVec double; % muA and muS for mesh
    % Gaussian error to add to the simulated data
    params.stdError = 0;
end

muAVec = xVec(1:end/2);
muSVec = xVec(end/2+1:end);
nMeasurements = sum(Geometry.measConfig > 0, 'all');
nFreqs = length(Geometry.freqsVec);
% 2 for log ampliture and phase
dataSim = zeros(nMeasurements * nFreqs, 2);

for freqIdx = 1:nFreqs
    freq = Geometry.freqsVec(freqIdx); % MHz
    % System matrix (nMeshNode, nMeshNode) for the forward problem, which
    % contains all the components of the DA in DOT
    sysMatrix = dotSysmat(Geometry.hMesh, muAVec, muSVec, Geometry.refIndVec, freq);
    % Solving the system of linear quations sysForwardMat * phi =
    % qvecSources, phi is the fluence
    fluence = sysMatrix \ Geometry.qVec;

    % Calculated the measured values at the detectors
    % .' = transpose; ' = complex conjugate transpose
    fluenceMeasured = Geometry.mVec.' * fluence;
    fluenceMeasured = fluenceMeasured(Geometry.hMesh.DataLinkList());
    fluenceMeasured = log(fluenceMeasured(:));
    dataSim((freqIdx - 1) * nMeasurements + 1:freqIdx * nMeasurements, :) ...
        = [real(fluenceMeasured(:)) unwrap(imag(fluenceMeasured(:)))];
end

dataSim = [dataSim(:, 1); dataSim(:, 2)];
fluenceSim = fluenceMeasured;

if params.stdError > 0
    dataSim = dataSim + params.stdError * randn(size(dataSim)) .* abs(dataSim);
end

end
