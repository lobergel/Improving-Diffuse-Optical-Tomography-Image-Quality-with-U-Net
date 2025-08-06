function xVecRecon = reconstructDifference(yMeasured, Geometry, Prior, params)
% reconstructDifference - difference reconstruction using amplidute data of
% yMeasured.
% 
% Arguments:
%   yMeasured       - difference data (phantoms with inclusions - phantom
%                     without inclusions)
%   Geometry        - structure containing the geometry of the model
%                     including toastMesh, toastBasis to map mesh to 
%                     regular pixel grid for forward model, qVec matrix 
%                     of sources, mVec matrix of detectors, refVec vector
%                     of refractive intices for each node in mesh, freqsVec
%                     vector of frequencies to be used for reconstruction
%   Prior           - structure containing priors: xVec is the vector
%                     of muA and muS priors, L1muA and L1muS are the
%                     Cholesky decomposition of inverse covariance matrix
%                     for muA and muS priors, hessian is part of the
%                     Hessian matrix containing covariance of the muA and
%                     muS priors needed for regularization
%   params.type     - type of AnalysisType
%   params.isGridBasis - if use grid basis instead of mesh
%
% konstantin.tamarov@uef.fi

arguments
    yMeasured double; % the measured or simulated boundary data
    Geometry struct; % structure describing the mesh and basis geometery
    Prior struct; % structure describing prior for reconstruction
    % type of reconstruction: only amplitude (absorption) or amplitude +
    % phase (absorption + scattering)
    params.type AnalysisType = AnalysisType.AbsorptionScattering;
    % use grid basis for reconstruction
    params.isGridBasis = 0; 
end

if params.isGridBasis
    nodeCount = Geometry.hBasis.slen;
    xVec = [Geometry.hBasis.Map('M->S', Prior.xVec(1:length(Prior.xVec)/2));
        Geometry.hBasis.Map('M->S', Prior.xVec(length(Prior.xVec)/2+1:end))];
else
    nodeCount = Geometry.hMesh.NodeCount();
    xVec = Prior.xVec;
end



cLight = 0.3 / mean(Geometry.refIndVec);
nFreqs = length(Geometry.freqsVec);

hBasis = 0;
if params.isGridBasis
    hBasis = Geometry.hBasis;
end

jacFreq = cell(nFreqs, 1);
for freqInd = 1:nFreqs
    jac = toastJacobian(Geometry.hMesh, hBasis, Geometry.qVec, Geometry.mVec, ...
        Prior.xVec(1:length(Prior.xVec)/2), ...
        Prior.xVec(length(Prior.xVec)/2+1:end), ...
        Geometry.refIndVec, Geometry.freqsVec(freqInd));
    jac(:, 1:nodeCount) = jac(:, 1:nodeCount) * cLight;
    jac(:, nodeCount+1:end) = jac(:, nodeCount+1:end) ...
        * diag(-cLight ./ (3 * (xVec(1:nodeCount) + xVec(nodeCount+1:end)).^2));
    jac(:, 1:nodeCount) = jac(:, 1:nodeCount) + jac(:, nodeCount+1:end);
    jacFreq{freqInd} = jac;
end
clear jac;
jacAmplitudes = []; jacPhases = [];
for freqInd = 1:nFreqs
    temp = jacFreq{freqInd};
    jacAmplitudes = [jacAmplitudes; temp(1:end/2, :)];
    jacPhases = [jacPhases; temp(end/2+1:end, :)];
end
jac = [jacAmplitudes; jacPhases];
clear jacFreq;

% Do the reconstruction
if params.type == AnalysisType.Absorption
    hessian  = [Prior.L1(1:end/2, 1:end/2) * (jac(1:end/2, :)); Prior.hessian];
    yData = [Prior.L1(1:end/2, 1:end/2) * (yMeasured(1:end/2)); zeros(2 * nodeCount, 1)]; %Lx*0*ones(2*n,1)];
elseif params.type == AnalysisType.AbsorptionScattering
    hessian  = [Prior.L1 * jac; Prior.hessian];
    yData = [Prior.L1 * yMeasured; zeros(2 * nodeCount, 1)]; %Lx*0*ones(2*n,1)];
else
    throw(MException('The reconstruction type has incorrect value.'));
end

xVecRecon = (hessian) \ (yData);

end

