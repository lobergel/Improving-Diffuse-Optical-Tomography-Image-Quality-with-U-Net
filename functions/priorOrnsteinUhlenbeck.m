function L1 = priorOrnsteinUhlenbeck(meshVertices, priorStd, inclusionSize, type)
% priorOrnsteinUhlenbeck - constructs the covariance matrix of prior 
%   described in Lieberman et al. 2010.
%
% Agruments:
%   meshVertices  - a (nMeshNodes, dimention of problem) matrix of mesh vertices
%                   coordinates obtained by calling toastMesh.Data() method
%   priorStd      - standard deviation of the prior
%   inclusionSize - prior inclusion size to calculate covariance; it is the
%                   distance that descrives how distantly located points in mesh
%                   relate to each other
%   
% Output:
%   L1 - a (nMeshNodes, nMeshNodes) Cholesky decomposed matrix of the
%        inverse covariance matrix; 
%
% A. Lipponen 11.9.2012
% Modified by K. Tamarov 8.2.2024 to handle arbitrary number of dimensions
%
% meghdoot.mozumder@uef.fi
% konstantin.tamarov@uef.fi

if isstruct(meshVertices)
    [meshVertices, ~, ~] = meshVertices.hMesh.Data();
end

nMeshNodes = size(meshVertices, 1);

% the (nVertices, nVertices) covariance matrix
covarMatrix = single(zeros(nMeshNodes));

for i=1:nMeshNodes
    % Rsq = sqrt((meshVtx(:, 1) - meshVtx(i, 1)).^2 + (meshVtx(:, 2) - meshVtx(i, 2)).^2);
    % Squared distance between the i point in mesh to all other points in
    % mesh
    rSq = sqrt(sum((meshVertices(:, :) - meshVertices(i, :)).^2, 2));
%     rSq = sqrt((meshVertices(:, 1) - meshVertices(i, 1)).^2 ...
%         + (meshVertices(:, 2) - meshVertices(i, 2)).^2 ...
%         + (meshVertices(:, 3) - meshVertices(i, 3)).^2);
    % Covariance describes how far different points in mesh are related to
    % each other.
    covarMatrix(i, :) = single(priorStd^2 * exp(-rSq / inclusionSize));
end

% Draw samples from the prior
if exist('type', 'var') && strcmp(type, 'sample')
    % Calculate Cholesky decomposition for the covariance matrix for sampling
    L1 = chol(covarMatrix, 'lower');
else
    % Get Cholesky decomposition of the precision matrix of prior (Upper
    % trianglular matrix)
    inverseCovarMatrix = inv(covarMatrix);
    clear covarMatrix;
    L1 = double(chol(inverseCovarMatrix));
end

end

