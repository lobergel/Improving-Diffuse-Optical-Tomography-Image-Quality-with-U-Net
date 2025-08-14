% Adjustable Parameters:
% - num_of_data: Number of data samples/images 
% - res: Image resolution (e.g., 32 for 32x32 images)
% - meshName: Filename for the mesh used
% - mua0, mus0: Baseline optical properties (absorption and scattering coefficients)

% Physical constraints:
% - min_mua, max_mua, min_mus, max_mus: Limits for optical properties, 
%   must be within physically meaningful ranges based on the mesh size

% Note:
% - The mesh dimensions are fixed at 80mm × 80mm for this version 
% - When creating validation data, change filenames ('mua_valid_recon.mat', etc.)

clear; close all;

res = 32;                                  % resolution of images (reconstructions, targets)
num_of_data = 5000;                        % number of images created in the for-loop

mua0 = 0.01; mus0 = 1;                     % background values for µa, µs' (mm^-1)

min_mua = 0.005; max_mua = 0.02;           % minimum and maximum µa coefficients 
min_mus = 0.5; max_mus = 2;                % minimum and maximum µs' coefficients 

min_noise = 0.005; max_noise = 0.03;       % minimum and maximum percentage for noise 

min_incl_radius = 3; max_incl_radius = 12; % minimum and maximum radius for the inclusions (in mm) 

min_x = -35; max_x = 35;                   % minimum and maximum x-coordinate for the inclusions (in mm)
min_y = -35; max_y = 35;                   % minimum and maximum y-coordinate for the inclusions (in mm)

filename_recon_mua = 'mua_recon.mat';      % filename for µa reconstructions 
filename_recon_mus = 'mus_recon.mat';      % filename for µs' reconstructions 
filename_target_mua = 'mua_target.mat';    % filename for µa targets 
filename_target_mus = 'mus_target.mat';    % filename for µs' targets 

rng('shuffle');                            % Used for-loop creates random variation of inclusions. With rng('shuffle'), 
                                           % for-loop doesn't create the same inclusions, as the loop is run again.
startTime = tic();

addpath('./functions/');
dataDir = './data/';
meshDir = './meshfiles/';

load([dataDir 'measConfig.mat']); % source-detector configurations 
load([dataDir 'stdError.mat']);

% createGeometry generates mesh, source-detector configurations, grid with wanted dimensions, etc. 
% createGeometry is used to have the reconstructions and eventual measurement-reconstructions be as similar as possible
Geometry = createGeometry(meshName = "mesh2D_r28.msh", meshDir = meshDir, isGmsh = 0, ...
    measConfig = measConfig, dataDir = dataDir, gridDimensions = [res res]);

hMesh = Geometry.hMesh;
hBasis = Geometry.hBasis;
qvec = Geometry.qVec;
mvec = Geometry.mVec;  
ref = Geometry.refIndVec;
DataLinkList = hMesh.DataLinkList();

[vtx, elem, ~] = hMesh.Data(); 
n = size(vtx, 1);              % number of nodes 

muareconSet_new = zeros(res, res, num_of_data);  % reconstructed µa
muspreconSet_new = zeros(res, res, num_of_data); % reconstructed µs'
muatargetSet_new = zeros(res, res, num_of_data); % target µa 
mustargetSet_new = zeros(res, res, num_of_data); % target µs'

fprintf('Progress: %5.1f%% (%5d / %5d)', 0, 0, num_of_data);
backspace_count = length(sprintf('%5.1f%% (%5d / %5d)', 0, 0, num_of_data));

for i = 1:num_of_data

    % Inclusion 1
    r1 = randi([min_incl_radius max_incl_radius]);                           % radius
    cx1 = min_x + (max_x - min_x) * rand();                                  % x-coordinate
    cy1 = min_y + (max_y - min_y) * rand();                                  % y-coodinate
    Index1 = find(sqrt((cx1 - vtx(:,1)).^2 + (cy1 - vtx(:,2)).^2) < r1);     % indeces corresponding to the position

    % Inclusion 2
    r2 = randi([min_incl_radius max_incl_radius]);                           % radius 
    cx2 = min_x + (max_x - min_x) * rand();                                  % x-coordinate
    cy2 = min_y + (max_y - min_y) * rand();                                  % y-coordinate
    Index2 = find(sqrt((cx2 - vtx(:,1)).^2 + (cy2 - vtx(:,2)).^2) < r2);     % indeces corresponding to the position

    mus = mus0 * ones(n,1);                                                  % initializing µs' values
    mua = mua0 * ones(n,1);                                                  % initializing µa values

    mus(Index1) = min_mus + (max_mus - min_mus) * rand();                    % adding inclusion 1 with random coefficient
    mua(Index2) = min_mua + (max_mua - min_mua) * rand();                    % adding inclusion 2 with random coefficient

    % Inclusion 3 
    if (randi([0 1]))                                                        % inclusion 3 is created only for some images 
        r3 = randi([min_incl_radius max_incl_radius]);                       % radius 
        cx3 = min_x + (max_x - min_x) * rand();                              % x-coordinate
        cy3 = min_y + (max_y - min_y) * rand();                              % y-coordinate
        Index3 = find(sqrt((cx3 - vtx(:,1)).^2 + (cy3 - vtx(:,2)).^2) < r3); % indeces corresponding to the position
        mua(Index3) = min_mua + (max_mua - min_mua) * rand();                % adding inclusion 3 with random coefficient
    end

    % Inclusion 4
    if (randi([0 1]))                                                        % inclusion 4 is created only for some images 
        r4 = randi([min_incl_radius max_incl_radius]);                       % radius 
        cx4 = min_x + (max_x - min_x) * rand();                              % x-coordinate
        cy4 = min_y + (max_y - min_y) * rand();                              % y-coordinate
        Index4 = find(sqrt((cx4 - vtx(:,1)).^2 + (cy4 - vtx(:,2)).^2) < r4); % indeces corresponding to position 
        mus(Index4) = min_mus + (max_mus - min_mus) * rand();                % adding inclusion 4 with random coefficient 
    end

    dmuatgt = mua - mua0;                                                    % µa inclusions map (difference from background)
    dmustgt = mus - mus0;                                                    % µs' inclusion map (difference from background) 

    freq = 100;                                                              % measurement frequency

    % Forward solve, with inclusions 
    K = dotSysmat(hMesh, mua, mus, ref, freq); 
    phi = K \ qvec;
    gamma = mvec.' * phi;
    gamma = gamma(DataLinkList());
    log_gamma = log(gamma(:));
    real_part = real(log_gamma);
    imag_part = unwrap(imag(log_gamma));
    
    y = [real_part; imag_part];

    % Forward solve, no inclusions (reference)
    mus_ref = mus0 * ones(n, 1);
    mua_ref = mua0 * ones(n, 1);
    K0 = dotSysmat(hMesh, mua_ref, mus_ref, ref, freq);
    phi0 = K0 \ qvec;
    gamma0 = mvec.' * phi0;
    gamma0 = gamma0(DataLinkList());
    log_gamma0 = log(gamma0(:));
    real_part0 = real(log_gamma0);
    imag_part0 = unwrap(imag(log_gamma0));
    
    y0 = [real_part0; imag_part0];

    deltay = y - y0;                                                       % difference from reference 

    noise = min_noise + (max_noise - min_noise) * rand();                  % percentage of noise 
    deltay = deltay + noise * randn(length(deltay), 1) .* abs(deltay);     % adding Gaussian noise to the difference data 

    % Solving the inverse problem 
    J = toastJacobian(hMesh, [], qvec, mvec, mua_ref, mus_ref, ref, freq); % Jacobian 
    stdn = stdError;                                                       % standard error from measurement data 
    Le = diag(1 ./ stdn);

    % Spatial prior, OU prior 
    r_prior = 8;
    prior_std_mua = mua0;                                                    
    prior_std_mus = mus0;                                                    
    Lxmua = PriorOrnsteinUhlenbeck(struct('g', vtx), prior_std_mua, r_prior);
    Lxmus = PriorOrnsteinUhlenbeck(struct('g', vtx), prior_std_mus, r_prior);
    Lx = [Lxmua sparse(n,n); sparse(n,n) Lxmus];

    % Solving the system 
    x = ([Le * J; Lx]) \ ([Le * deltay; zeros(2 * n, 1)]);
    muarecon = x(1:n);
    musprecon = x(n+1:end);

    % Mapping from mesh to grid 
    muareconGrid = rot90(reshape(hBasis.Map('M->B', muarecon), [res, res]), 1);    % contains µa reconstruction
    muspreconGrid = rot90(reshape(hBasis.Map('M->B', musprecon), [res, res]), 1);  % contains µs' reconstruction
    muatargetGrid = rot90(reshape(hBasis.Map('M->B', dmuatgt), [res, res]), 1);    % contains µa target
    mustargetGrid = rot90(reshape(hBasis.Map('M->B', dmustgt), [res, res]), 1);    % contains µs' target 

    muareconSet_new(:, :, i) = muareconGrid;                                       % adds µa reconstruction to a temporary matrix 
    muspreconSet_new(:, :, i) = muspreconGrid;                                     % adds µs' reconstruction to a temporary matrix
    muatargetSet_new(:, :, i) = muatargetGrid;                                     % adds µa target to a temporary matrix
    mustargetSet_new(:, :, i) = mustargetGrid;                                     % adds µs' target to a temporary matrix

    % Progress counter 
    percent = 100 * i/num_of_data;
    progress_str = sprintf(' %5.1f%% (%5d / %5d)', percent, i, num_of_data);
    fprintf(['\b' repmat('\b', 1, backspace_count) '%s'], progress_str);
end
    
%% Saving data to files 

% After the for-loop, the existence of files is checked. If the files do not exist, they are created.
if isfile(filename_recon_mua)
    load(filename_recon_mua, 'muareconSet');
else
    muareconSet = [];
end

if isfile(filename_recon_mus)
    load(filename_recon_mus, 'muspreconSet');
else
    muspreconSet = [];
end

if isfile(filename_target_mua)
    load(filename_target_mua, 'muatargetSet');
else
    muatargetSet = [];
end

if isfile(filename_target_mus)
    load(filename_target_mus, 'mustargetSet');
else
    mustargetSet = [];
end

% Appending newly created matrices to any existing dataset in file
muareconSet = cat(3, muareconSet, muareconSet_new);
muspreconSet = cat(3, muspreconSet, muspreconSet_new);
muatargetSet = cat(3, muatargetSet, muatargetSet_new);
mustargetSet = cat(3, mustargetSet, mustargetSet_new);

save(filename_recon_mua, 'muareconSet');
save(filename_recon_mus, 'muspreconSet');
save(filename_target_mua, 'muatargetSet');
save(filename_target_mus, 'mustargetSet');

% Summary is displayed. All files should have the same amount of matrices. Additionally, the elapsed time is displayed. 
fprintf('\nSaved data:\n');
fprintf('  %s: %d reconstructions\n', filename_recon_mua,  size(muareconSet, 3)); 
fprintf('  %s: %d reconstructions\n', filename_recon_mus,  size(muspreconSet, 3));
fprintf('  %s: %d target images\n', filename_target_mua, size(muatargetSet, 3));
fprintf('  %s: %d target images\n', filename_target_mus, size(mustargetSet, 3));

elapsedTime = toc(startTime);
hours = floor(elapsedTime / 3600);
minutes = floor(mod(elapsedTime, 3600) / 60);
seconds = mod(elapsedTime, 60);

if (hours ~= 0)
    fprintf('Elapsed time: %d hour(s), %d minute(s), %.f second(s)\n', hours, minutes, seconds);
else
    fprintf('Elapsed time: %d minute(s), %.f second(s)\n', minutes, seconds);
end
