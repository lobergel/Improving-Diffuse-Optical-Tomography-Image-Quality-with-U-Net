close; clear all;

N = 16; % number of sources (and number of detectors) in range [0, 2pi]
res = 32; % resolution of output grid (res x res)
numOfData = 1000; % number of iteration in the for-loop (how many res x res matrices are created)

% filenames for reconstruction and target matrices 
filename_mua_recon = '360_mua_recon.mat';
filename_mus_recon = '360_mus_recon.mat';
filename_mua_target = '360_mua_target.mat';
filename_mus_target = '360_mus_target.mat';

ths = (0 : 2*pi/N : 2 * pi)'; % source angles 
sp = [25*cos(ths) 25*sin(ths)]; % source positions in circle

thd = (pi/N : 2*pi/N : 2 * pi)'; % detector angles with offset 
mp = [25*cos(thd) 25*sin(thd)]; % detector positions in circle 

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

% check and load existing files if they exist (prevents overwriting) 
if isfile(filename_mua_recon)
    load(filename_mua_recon, 'muareconSet');
    nExistingMua = size(muareconSet, 3);
else
    muareconSet = [];
    nExistingMua = 0;
end

if isfile(filename_mus_recon)
    load(filename_mus_recon, 'muspreconSet');
    nExistingMus = size(muspreconSet, 3);
else
    muspreconSet = [];
    nExistingMus = 0;
end

if isfile(filename_mua_target)
    load(filename_mua_target, 'muatargetSet');
    nExistingMuaTarget = size(muatargetSet, 3);
else
    muatargetSet = [];
    nExistingMuaTarget = 0;
end

if isfile(filename_mus_target)
    load(filename_mus_target, 'mustargetSet');
    nExistingMusTarget = size(mustargetSet, 3);
else
    mustargetSet = [];
    nExistingMusTarget = 0;
end

%% loop 

rng('shuffle');

for i = 1:numOfData    
    % load mesh data
    [vtx, elem, ~] = hMesh.Data; % extracting mesh data
    n = size(vtx, 1); % node count

    % inclusion 1 (scattering)
    r1 = randi([3 12]); % integer radius for inclusion 1, range [3, 12]
    cx1 = -18 + 36 * rand(); % x-coordinate for inclusion 1, range [-18, 18]
    cy1 = -18 + 36 * rand(); % y-coordinate for inclusion 1, range [-18, 18]
    Index1 = find(sqrt((cx1 - vtx(:,1)).^2 + (cy1 - vtx(:,2)).^2) < r1); % indexes from mus that correspond with inclusion 1

    % inclusion 2 (absorption)
    r2 = randi([3 12]); % [3,12]
    cx2 = -18 + 36 * rand(); % [-18, 18]
    cy2 = -18 + 36 * rand(); % [-18, 18]
    Index2 = find(sqrt((cx2 - vtx(:,1)).^2 + (cy2 - vtx(:,2)).^2) < r2);

    % inclusion 3 (another absorption)
    r3 = randi([3 12]);
    cx3 = -18 + 36 * rand();
    cy3 = -18 + 36 * rand();
    Index3 = find(sqrt((cx3 - vtx(:,1)).^2 + (cy3 - vtx(:,2)).^2) < r3);

    % inclusion 4 (another scattering)
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

    % random percentage of Gaussian noise 
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

    % appending matrices 
    musreconSet(:, :, nExistingMua + i) = muareconGrid;
    muspreconSet(:, :, nExistingMus + i) = muspreconGrid;
    muatargetSet(:, :, nExistingMuaTarget + i) = muatargetGrid;
    mustargetSet(:, :, nExistingMusTarget + i) = mustargetGrid;

    fprintf('Iteration %d completed.\n', i); % (optional line) 
end

% save all reconstructions and targets
save(filename_mua_recon, 'musreconSet');
save(filename_mus_recon, 'muspreconSet');
save(filename_mua_target, 'muatargetSet');
save(filename_mus_target, 'mustargetSet');

%% testing the set of data (optional)
numImages = size(musreconSet, 3);
fprintf("Files have %d reconstructions.\n", numImages); % all files should have the same amount of matrices 

% loop displays 4 random reconstructions and their targets from the newest set of reconstructions 
figure;
for i = 1:4
    randIndex = randi([numImages-numOfData numImages]); % index value from the newest reconstructions 

    subplot(4, 4, (i-1)*4 + 1); imagesc(musreconSet(:, :, randIndex)); axis equal tight off; colorbar; colormap(1 - gray); title(['\delta\mu_a Sample ' num2str(randIndex)]);
    subplot(4, 4, (i-1)*4 + 2); imagesc(muatargetSet(:, :, randIndex)); axis equal tight off; colorbar; colormap(1 - gray); title(['\delta\mu_a Target ' num2str(randIndex)]);
    subplot(4, 4, (i-1)*4 + 3); imagesc(muspreconSet(:, :, randIndex)); axis equal tight off; colorbar; colormap(1 - gray); title(['\delta\mu_s Sample ' num2str(randIndex)]); 
    subplot(4, 4, (i-1)*4 + 4); imagesc(mustargetSet(:, :, randIndex)); axis equal tight off; colorbar; colormap(1 - gray); title(['\delta\mu_s Target ' num2str(randIndex)]);
end

%% saving global min/max values for visualization scaling

filename_scale_mua = 'mua_scale';
filename_scale_mus = 'mus_scale';

mua_max = max([muareconSet(:); muatargetSet(:)]); 
mua_min = min([muareconSet(:); muatargetSet(:)]);

mus_max = max([muspreconSet(:); mustargetSet(:)]);
mus_min = min([muspreconSet(:); mustargetSet(:)]);

save(filename_scale_mua, "mua_max", "mua_min");
save(filename_scale_mus, "mus_max", "mus_min"); 
   
