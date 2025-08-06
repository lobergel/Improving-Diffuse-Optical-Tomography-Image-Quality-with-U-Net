function Geometry = createGeometry(params)
% loadGeometry - prepares experiment geometry by loading mesh, setting up
% sources and detectors, frequencies, refractive index, etc.
%   Works for the cylindrical phantoms used in the time-domain DOT setup

arguments
    % you can supply the old Geometry and update parts of it (mainly needed
    % not to reload mesh itself)
    params.geometry struct = struct();
    params.meshDir string = "../meshes/";
    params.meshName string = "dot-mid-n6686.msh";
    params.isGmsh double = 1;
    % folder name with experiment data, the default value is for
    % configuration used for simulation
    params.dataDir string = "../data/SimulationExp";
    % CSV file contains the configuration of active source-detector pairs
    params.csvFile string = "separate_source_det_rings.csv";
    % matrix with measurement configuration (usually obtained from loadData
    % function)
    params.measConfig double = 0;
    % Minimum and maximum coordinates in millimeters to which the mesh will
    % be rescaled
    params.sizesMin (1,:) double = [-40 -40 -50]; % mm
    params.sizesMax (1,:) double = [40 40 50]; % mm
    % grid dimensions, if zero, the size will be 1 grid element per mm
    params.gridDimensions = 0;
    % frequency vector in MHz, can contain several freqs
    params.freqsVec double = 56.980056980056460; % MHz
    params.refInd double {mustBePositive} = 1.33;
    % z coordinates of source and detector rings
    % sources are +5 mm from the geometry center
    params.zSources double = 5; % mm
    % detectors are -5 mm from the geometry center
    params.zDetectors double = -5; % mm
    % source paramaters
    params.sourceType string = "Neumann";
    params.sourceProfile string ="Gaussian";
    % source width can be single nonzero value or the array of widths
    % equal to the number of sources (see CSV file)
    params.sourceWidth double {mustBePositive} = 8; % mm
    % detector parameters
    params.detectorProfile string = "Gaussian";
    params.detectorWidth double {mustBePositive} = 0.6; % mm
    params.detectorRefInd double {mustBePositive} = 1.33;

    params.plotMesh = 0;
end

if isempty(fieldnames(params.geometry))
    % load the mesh from indicated file
    Geometry = struct();
    Geometry.meshPath = [what(params.meshDir).path '/' char(params.meshName)];
    if params.isGmsh
        Geometry.hMesh = toastMesh(Geometry.meshPath, 'gmsh');
    else
        Geometry.hMesh = toastMesh(Geometry.meshPath);
    end
else
    Geometry = params.geometry;
end

if length(params.measConfig) < 2
    csvFile = [what(params.dataDir).path '/' char(params.dataName) '/' char(params.csvFile)];
    params.measConfig = readmatrix(csvFile);
end

% Geometry.dataPath = char(params.dataPath);
% if strlength(Geometry.dataPath) == 0
%     Geometry.dataPath = [what(params.dataDir).path '/' char(params.dataName) '/'];
%     if strlength(params.csvFile) == 0
%         csvFiles = dir([Geometry.dataPath '*.csv']);
%         params.csvFile = csvFiles.name(1);
%     end
% end
% 
% % load the configuration of active source-detector pairs
% if ~contains(Geometry.dataPath, "SimulationExp")
%     csvFiles = dir([Geometry.dataPath '*.csv']);
%     params.csvFile = csvFiles(1).name;
% end

Geometry.dataDir = char(params.dataDir);
[Geometry.measConfig, Geometry.nSources, Geometry.nDetectors] = ...
    binaryMeasConfig(params.measConfig);

% updating the mesh sizes
sizes = params.sizesMax - params.sizesMin;
if sizes(1) ~= sizes(2)
    MException("Mesh size: x and y sizes are not equal.", ...
        "x and y sizes for the mesh must be equal for cylindrical geometry");
end
Geometry.sizes = sizes;
[vtx, idx, eltp] = Geometry.hMesh.Data(); % see Toast docs
for i = 1:size(vtx, 2) % see Toast docs
    % loop through x, y and (if exists) z coords for all vertices
    vtx(:, i) = (params.sizesMin(i) + params.sizesMax(i)) / 2 ...
        + (params.sizesMax(i) - params.sizesMin(i)) ...
        ./ (max(vtx(:, i)) - min(vtx(:, i))) .* vtx(:, i);
end
Geometry.hMesh = toastMesh(vtx, idx, eltp);
meshDim = size(vtx, 2);
Geometry.sizes = Geometry.sizes(1:meshDim);
clear vtx idx eltp;

% frequencies and refractive index
Geometry.freqsVec = params.freqsVec;
Geometry.refIndVec = params.refInd * ones(Geometry.hMesh.NodeCount(), 1);

% location of sources and detectors on the mesh and for the problem
% coordinates of sources and detectors
% for alternating sources-detectors in one ring
% sources are +5 mm from the geometry center
cCoords = (params.sizesMax + params.sizesMin) ./ 2;
if (meshDim == 2)
    cCoords = [cCoords 0];
end
cCoords(3) =  cCoords(3) + params.zSources;
Geometry.sourceCoords = optodesRing(Geometry.nSources, Geometry.sizes(1)/2, cCoords, 0);
Geometry.sourceCoords = Geometry.sourceCoords(:, 1:meshDim);

cCoords(3) = cCoords(3) - params.zSources + params.zDetectors;
Geometry.detectorCoords = optodesRing(Geometry.nDetectors, Geometry.sizes(1)/2, ...
    cCoords, - (2 * pi / Geometry.nDetectors) / 2);
Geometry.detectorCoords = Geometry.detectorCoords(:, 1:meshDim);

% Keep parameters of sources and detectors
Geometry.sourceType = char(params.sourceType);
Geometry.sourceProfile = char(params.sourceProfile);
Geometry.sourceWidth = params.sourceWidth;
Geometry.detectorProfile = char(params.detectorProfile);
Geometry.detectorWidth = params.detectorWidth;
Geometry.detectorRefInd = params.detectorRefInd;
% Create sources and detectors on the mesh
% Save the source and detector configuration to file
% Check if the mesh is 3D or 2D and update source and detector coords

Geometry.hMesh.WriteQM([Geometry.dataDir 'QMfile.txt'], Geometry.sourceCoords(:, 1:meshDim), ...
    Geometry.detectorCoords(:, 1:meshDim), sparse(Geometry.measConfig));
% Setup the source and detectors on the mesh
Geometry.hMesh.ReadQM([Geometry.dataDir 'QMfile.txt']);
% See Toast help for parameter explanation
Geometry.qVec = Geometry.hMesh.Qvec(Geometry.sourceType, Geometry.sourceProfile, Geometry.sourceWidth);
Geometry.mVec = Geometry.hMesh.Mvec(Geometry.detectorProfile, Geometry.detectorWidth, Geometry.detectorRefInd);

% Create basis to map mesh to regular pixel grid
if sum(params.gridDimensions) > 0
    basisSize = params.gridDimensions;
else
    % one grid pixel/voxel for 1 mm
    basisSize = Geometry.sizes;
end
Geometry.hBasis = toastBasis(Geometry.hMesh, basisSize, 'Linear');
Geometry.dims = Geometry.hBasis.Dims();
Geometry.gridCornerCoords = {...
    linspace(-sizes(1)/2, sizes(1)/2, Geometry.dims(1)+1)' ...
    linspace(-sizes(2)/2, sizes(2)/2, Geometry.dims(2)+1)'
    };
if meshDim == 3
    Geometry.gridCornerCoords = [Geometry.gridCornerCoords ...
        linspace(-sizes(3)/2, sizes(3)/2, Geometry.dims(3)+1)'];
end
Geometry.gridCenterCoords = cell(1, 3);
for i = 1:length(Geometry.gridCornerCoords)
    Geometry.gridCenterCoords{i} = (Geometry.gridCornerCoords{i}(2:end) ...
        - Geometry.gridCornerCoords{i}(1:end-1)) / 2 ...
        + Geometry.gridCornerCoords{i}(1:end-1);
end

if params.plotMesh
    if meshDim == 2
        figure; toastShowMesh(Geometry.hMesh); hold on;
        scatter(Geometry.sourceCoords(:, 1), Geometry.sourceCoords(:, 2), 100, "red", "filled", "o");
        scatter(Geometry.detectorCoords(:, 1), Geometry.detectorCoords(:, 2), 100, "green", "filled", "o");
        xlabel('X [mm]'); ylabel("Y [mm]");
    end
    if meshDim == 3
        figure; toastShowMesh(Geometry.hMesh); hold on;
        scatter3(Geometry.sourceCoords(:, 1), Geometry.sourceCoords(:, 2), ...
            Geometry.sourceCoords(:, 3), 100, "red", "filled", "o");
        scatter3(Geometry.detectorCoords(:, 1), Geometry.detectorCoords(:, 2), ...
            Geometry.detectorCoords(:, 3), 100, "green", "filled", "o");
        xlabel('X [mm]'); ylabel("Y [mm]"); zlabel("Z [mm]");
    end
end

end

