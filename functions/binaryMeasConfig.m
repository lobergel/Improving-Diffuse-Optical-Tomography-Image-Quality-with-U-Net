function [binaryMeasConfig,nSources,nDetectors] = binaryMeasConfig(measConfig)
% binaryMeasConfig - generates the binary measurement configuration file
% for Toast out of the CSV configuration file.
% 
% Arguments:
%   measConfig  - matrix with source-detector configuration (gor from
%                 reading CSV measurement xonfic file)
%
% Output:
%   measConfig  - Toast binary measurement config listing which detector
%                 are active for each source
%   nSources    - number of sources
%   nDetectors  - number of detectors
%
% konstantin.tamarov@uef.fi

nSources = size(measConfig, 1);
nDetectors = max(measConfig, [], "all");
binaryMeasConfig = zeros(nSources, nDetectors);
for ind = 1:nSources
    % loop is needed in case the binary matrix is not square
    binaryMeasConfig(ind, measConfig(ind, measConfig(ind, :) > 0)) = 1;
end
binaryMeasConfig = binaryMeasConfig';
end

