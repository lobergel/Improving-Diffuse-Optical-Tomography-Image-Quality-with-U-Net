close all; clear all;

N = 16; % number of sources (and number of detectors) in range [0, 2pi]
res = 32; % resolution for reconstructions

% filenames for reconstruction and target matrices 
filename_mua_recon = 'diff_MuaReconstructedData.mat';
filename_mus_recon = 'diff_MusReconstructedData.mat';
filename_mua_target = 'diff_MuaTargetData.mat';
filename_mus_target = 'diff_MusTargetData.mat';

ths = (0 : 2*pi/N : 2 * pi)'; % range for sources
sp = [25*cos(ths) 25*sin(ths)]; % sources on unit circle with radius 25 

thd = (pi/N : 2*pi/N : 2 * pi)'; % range for detectors 
mp = [25*cos(thd) 25*sin(thd)]; % detectors on unit circle with radius 25 

% setting up a mesh 
meshname = 'mesh2D_r28.msh';
meshpath = 'meshfiles/';
hMesh = toastMesh([meshpath meshname]);
hMesh.SetQM(sp, mp);

% basis for mapping reconstructions to 2D grids from mesh
hBasis = toastBasis(hMesh, [res res], 'Linear');

% FEM setup
qvec = hMesh.Qvec('Neumann', 'Gaussian', 2);
mvec = hMesh.Mvec('Gaussian', 2, 1.4);
ref = ones(hMesh.NodeCount(), 1) * 1.4;

% number of iteration in the for-loop (how many res x res matrices are created)
numOfData = 1000;

% check and load existing files if they exist (prevents overwriting) 
if isfile(filename_mua_recon)
    load(filename_mua_recon, 'muareconMatrix');
    nExistingMua = size(muareconMatrix, 3);
else
    muareconMatrix = [];
    nExistingMua = 0;
end

if isfile(filename_mus_recon)
    load(filename_mus_recon, 'muspreconMatrix');
    nExistingMus = size(muspreconMatrix, 3);
else
    muspreconMatrix = [];
    nExistingMus = 0;
end

if isfile(filename_mua_target)
    load(filename_mua_target, 'muatargetMatrix');
    nExistingMuaTarget = size(muatargetMatrix, 3);
else
    muatargetMatrix = [];
    nExistingMuaTarget = 0;
end

if isfile(filename_mus_target)
    load(filename_mus_target, 'mustargetMatrix');
    nExistingMusTarget = size(mustargetMatrix, 3);
else
    mustargetMatrix = [];
    nExistingMusTarget = 0;
end

%% loop 
for i = 1:numOfData
    % load mesh data
    [vtx, elem, ~] = hMesh.Data;
    n = size(vtx, 1);

    % random inclusion 1 (scattering)
    r1 = randi([3 12]); % [3, 12]
    cx1 = -18 + 36 * rand(); % [-18, 18]
    cy1 = -18 + 36 * rand(); % [-18, 18]
    Index1 = find(sqrt((cx1 - vtx(:,1)).^2 + (cy1 - vtx(:,2)).^2) < r1);

    % Random inclusion 2 (absorption)
    r2 = randi([3 12]); % [3,12]
    cx2 = -18 + 36 * rand(); % [-18, 18]
    cy2 = -18 + 36 * rand(); % [-18, 18]
    Index2 = find(sqrt((cx2 - vtx(:,1)).^2 + (cy2 - vtx(:,2)).^2) < r2);

    % Random inclusion 3 (another absorption)
    r3 = randi([3 12]);
    cx3 = -18 + 36 * rand();
    cy3 = -18 + 36 * rand();
    Index3 = find(sqrt((cx3 - vtx(:,1)).^2 + (cy3 - vtx(:,2)).^2) < r3);

    % Random inclusion 4 (another scattering)
    r4 = randi([3 12]);
    cx4 = -18 + 36 * rand();
    cy4 = -18 + 36 * rand();
    Index4 = find(sqrt((cx4 - vtx(:,1)).^2 + (cy4 - vtx(:,2)).^2) < r4);

    % background values for mua, mus
    mua0 = 0.01; 
    mus0 = 1;

    mus = mus0 * ones(n,1);
    mua = mua0 * ones(n,1);

    % adding the inclusions in the appropriate locations with random coefficients from an appropriate range 
    mus(Index1) = 0.5 + (1.5 - 0.5) * rand(); % [0.5, 1.5]
    mus(Index4) = 0.5 + (1.5 - 0.5) * rand(); % [0.5, 1.5]

    mua(Index2) = 0.005 + (0.05 - 0.005) * rand(); % [0.005, 0.05]
    mua(Index3) = 0.005 + (0.05 - 0.005) * rand(); % [0.005, 0.05]

    dmuatgt = mua - mua0;
    dmustgt = mus - mus0;

    % generating measurement with inclusions
    freq = 100; % MHz
    K = dotSysmat(hMesh, mua, mus, ref, freq);
    phi = K \ qvec;
    gamma = mvec.' * phi;
    y = [real(log(gamma(:))); imag(log(gamma(:)))];

    % reference measurement (no inclusions)
    mus_ref = mus0 * ones(n,1);
    mua_ref = mua0 * ones(n,1);
    K0 = dotSysmat(hMesh, mua_ref, mus_ref, ref, freq);
    phi0 = K0 \ qvec;
    gamma0 = mvec.' * phi0;
    y0 = [real(log(gamma0(:))); imag(log(gamma0(:)))];

    deltay = y - y0;

    % random Gaussian noise 
    noise = 0.005 + (0.015 - 0.005) * rand(); % [0.005, 0.015]
    deltay = deltay + noise * randn(length(deltay), 1) .* abs(deltay);

    % building Jacobian and prior
    J = toastJacobian(hMesh, [], qvec, mvec, mua_ref, mus_ref, ref, freq);
    stdn = noise * abs(y);
    Le = diag(1 ./ stdn);

    r_prior = 8;
    prior_std_mua = mua0 / 10;
    prior_std_mus = mus0 / 10;
    Lxmua = PriorOrnsteinUhlenbeck(struct('g', vtx), prior_std_mua, r_prior);
    Lxmus = PriorOrnsteinUhlenbeck(struct('g', vtx), prior_std_mus, r_prior);
    Lx = [Lxmua sparse(n,n); sparse(n,n) Lxmus];

    x = ([Le * J; Lx]) \ ([Le * deltay; zeros(2 * n, 1)]);
    muarecon = x(1:n);
    musprecon = x(n+1:end);

    % mapping to 2D image (mesh to grid)
    muareconGrid = rot90(reshape(hBasis.Map('M->B', muarecon), [res, res]), 1);
    muspreconGrid = rot90(reshape(hBasis.Map('M->B', musprecon), [res, res]), 1);
    muatargetGrid = rot90(reshape(hBasis.Map('M->B', dmuatgt), [res, res]), 1);
    mustargetGrid = rot90(reshape(hBasis.Map('M->B', dmustgt), [res, res]), 1);

    muareconMatrix(:, :, nExistingMua + i) = muareconGrid;
    muspreconMatrix(:, :, nExistingMus + i) = muspreconGrid;
    muatargetMatrix(:, :, nExistingMuaTarget + i) = muatargetGrid;
    mustargetMatrix(:, :, nExistingMusTarget + i) = mustargetGrid;

    fprintf('Iteration %d completed.\n', i);
end

% save all reconstructions and targets
save(filename_mua_recon, 'muareconMatrix');
save(filename_mus_recon, 'muspreconMatrix');
save(filename_mua_target, 'muatargetMatrix');
save(filename_mus_target, 'mustargetMatrix');

%% testing the set of data (optional)
% figure;
numImages = size(muareconMatrix, 3);
fprintf("File %s has %d reconstructions.\n", filename_mua_recon, numImages); % all files should have the same amount of matrices 

% loop displays 4 random reconstructions and their targets from the newest set of reconstructions 
for i = 1:4
    randIndex = randi([numImages-numOfData numImages]); % index value from the newest reconstructions 

    subplot(4, 4, (i-1)*4 + 1); 
    imagesc(muareconMatrix(:, :, randIndex));
    axis equal tight off;
    colorbar;
    colormap(1 - gray);
    title(['\delta\mu_a Sample ' num2str(randIndex)]);

    subplot(4, 4, (i-1)*4 + 2); 
    imagesc(muatargetMatrix(:, :, randIndex));
    axis equal tight off;
    colorbar;
    colormap(1 - gray);
    title(['\delta\mu_a Target ' num2str(randIndex)]);

    subplot(4, 4, (i-1)*4 + 3); 
    imagesc(muspreconMatrix(:, :, randIndex));
    axis equal tight off;
    colorbar;
    colormap(1 - gray);
    title(['\delta\mu_s Sample ' num2str(randIndex)]);

    subplot(4, 4, (i-1)*4 + 4);
    imagesc(mustargetMatrix(:, :, randIndex));
    axis equal tight off;
    colorbar;
    colormap(1 - gray);
    title(['\delta\mu_s Target ' num2str(randIndex)]);
end

