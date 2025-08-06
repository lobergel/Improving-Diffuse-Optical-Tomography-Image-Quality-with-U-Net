close all; clear;

% sources and detectors are aligned evenly around a circle with the radius 25

N = 16; % number of sources and detectors (16s16d)
res = 32; % grid resolution
numOfData = 100; % number of created images 
radius = 25; % radius of source-detector circle

% temporary files
filename_mua_recon = 'parallel_mua_recon.mat';
filename_mus_recon = 'parallel_mus_recon.mat';
filename_mua_target = 'parallel_mua_target.mat';
filename_mus_target = 'parallel_mus_target.mat';

% loading existing data
if isfile(filename_mua_recon)
    load(filename_mua_recon, 'muareconSet');
else
    muareconSet = [];
end

if isfile(filename_mus_recon)
    load(filename_mus_recon, 'muspreconSet');
else
    muspreconSet = [];
end

if isfile(filename_mua_target)
    load(filename_mua_target, 'muatargetSet');
else
    muatargetSet = [];
end

if isfile(filename_mus_target)
    load(filename_mus_target, 'mustargetSet');
else
    mustargetSet = [];
end

% preallocate output arrays for parallel loop
muareconSet_new = zeros(res, res, numOfData);
muspreconSet_new = zeros(res, res, numOfData);
muatargetSet_new = zeros(res, res, numOfData);
mustargetSet_new = zeros(res, res, numOfData);

% source positions 
ths = (0 : 2*pi/N : 2 * pi)';
sp = [radius * cos(ths) radius * sin(ths)];

% detector positions 
thd = (pi/N : 2*pi/N : 2 * pi)';
mp = [radius * cos(thd) radius * sin(thd)];

% loading mesh and setting sources, detectors to positions 
meshname = 'mesh2D_r28.msh';
meshpath = 'meshfiles/';
hMesh = toastMesh([meshpath meshname]);
hMesh.SetQM(sp, mp);
hBasis = toastBasis(hMesh, [res res], 'Linear');

% FEM setup
qvec = hMesh.Qvec('Neumann', 'Gaussian', 2); % <- 
mvec = hMesh.Mvec('Gaussian', 2, 1.4);       % <-
ref = ones(hMesh.NodeCount(), 1) * 1.4;      % <- 

% start parallel pool
if isempty(gcp('nocreate'))
    parpool;
end

% minimum and maximum coefficient values for absorption and scattering 
mua_min = 0.005; mua_max = 0.02;
mus_min = 0.5; mus_max = 1.5; 

% background values 
mua0 = 0.01; mus0 = 1;

% minimum and maximum percentage of noise 
noise_min = 0.005; noise_max = 0.015; 

% minimum and maximum radius for inclusions 
radius_min = -18; radius_max = 18; % 50

startTime = tic; 

parfor i = 1:numOfData
    % local variables for workers 
    hMesh_local = toastMesh([meshpath meshname]);
    hMesh_local.SetQM(sp, mp);
    hBasis_local = toastBasis(hMesh_local, [res res], 'Linear');

    % loading mesh data
    [vtx, elem, ~] = hMesh_local.Data;
    n = size(vtx, 1);
    mus = mus0 * ones(n,1);
    mua = mua0 * ones(n,1);

    % inclusion 1 
    r1 = randi([3 12]);                                                  % radius 
    cx1 = radius_min + (radius_max - radius_min) * rand();               % x-position 
    cy1 = radius_min + (radius_max - radius_min) * rand();               % y-position 
    Index1 = find(sqrt((cx1 - vtx(:,1)).^2 + (cy1 - vtx(:,2)).^2) < r1); % node indexes 
    mus(Index1) = mus_min + (mus_max - mus_min) * rand();                % assigning a random scattering coefficient value 

    % inclusion 2 
    r2 = randi([3 12]);                                                  % radius 
    cx2 = radius_min + (radius_max - radius_min) * rand();               % x-coordinate
    cy2 = radius_min + (radius_max - radius_min) * rand();               % y-coordinate 
    Index2 = find(sqrt((cx2 - vtx(:,1)).^2 + (cy2 - vtx(:,2)).^2) < r2); % node indexes 
    mua(Index2) = mua_min + (mua_max - mua_min) * rand();                % assigning a random absorbing coefficient value 

    % inclusion 3, created for some iterations 
    if (randi([0 1]))
        r3 = randi([3 12]);                                                  % radius 
        cx3 = radius_min + (radius_max - radius_min) * rand();               % x-coordinate
        cy3 = radius_min + (radius_max - radius_min) * rand();               % y-coordinate
        Index3 = find(sqrt((cx3 - vtx(:,1)).^2 + (cy3 - vtx(:,2)).^2) < r3); % node indexes 
        mua(Index3) = mua_min + (mua_max - mua_min) * rand();                % assigning a random absorbing coefficient
    end

    % inclusion 4, created for some iterations 
    if (randi([0 1]))
        r4 = randi([3 12]);                                                  % radius 
        cx4 = radius_min + (radius_max - radius_min) * rand();               % x-coordinate
        cy4 = radius_min + (radius_max - radius_min) * rand();               % y-coordinate 
        Index4 = find(sqrt((cx4 - vtx(:,1)).^2 + (cy4 - vtx(:,2)).^2) < r4); % node indexes 
        mus(Index4) = mus_min + (mus_max - mus_min) * rand();                % assigning a random scattering coefficient 
    end

    % difference from background values 
    dmuatgt = mua - mua0;
    dmustgt = mus - mus0;

    % creating measurements with inclusions 
    freq = 100; 
    K = dotSysmat(hMesh_local, mua, mus, ref, freq);
    phi = K \ qvec;
    gamma = mvec.' * phi;
    y = [real(log(gamma(:))); imag(log(gamma(:)))];

    % measurements with no inclusion 
    mus_ref = mus0 * ones(n,1);
    mua_ref = mua0 * ones(n,1);
    K0 = dotSysmat(hMesh_local, mua_ref, mus_ref, ref, freq);
    phi0 = K0 \ qvec;
    gamma0 = mvec.' * phi0;
    y0 = [real(log(gamma0(:))); imag(log(gamma0(:)))];

    % difference between measurements with and without inclusions 
    deltay = y - y0;

    % adding random percentage of Gaussian noise 
    noise = noise_min + (noise_max - noise_min) * rand();
    deltay = deltay + noise * randn(length(deltay), 1) .* abs(deltay);

    % building Jacobian 
    J = toastJacobian(hMesh_local, [], qvec, mvec, mua_ref, mus_ref, ref, freq);
    stdn = noise * abs(y);
    Le = diag(1 ./ stdn); % L1

    % Ornstein-Uhlenbeck prior 
    r_prior = 8;
    prior_std_mua = mua0;
    prior_std_mus = mus0;
    Lxmua = PriorOrnsteinUhlenbeck(struct('g', vtx), prior_std_mua, r_prior);
    Lxmus = PriorOrnsteinUhlenbeck(struct('g', vtx), prior_std_mus, r_prior);
    Lx = [Lxmua sparse(n,n); sparse(n,n) Lxmus];

    % solving the inverse problem 
    x = ([Le * J; Lx]) \ ([Le * deltay; zeros(2 * n, 1)]);
    muarecon = x(1:n);
    musprecon = x(n+1:end);

    % mapping coefficients from mesh to grid and rotating 
    muareconGrid = rot90(reshape(hBasis_local.Map('M->B', muarecon), [res, res]), 1);
    muspreconGrid = rot90(reshape(hBasis_local.Map('M->B', musprecon), [res, res]), 1);
    muatargetGrid = rot90(reshape(hBasis_local.Map('M->B', dmuatgt), [res, res]), 1);
    mustargetGrid = rot90(reshape(hBasis_local.Map('M->B', dmustgt), [res, res]), 1);

    % appending matrices 
    muareconSet_new(:, :, i) = muareconGrid;
    muspreconSet_new(:, :, i) = muspreconGrid;
    muatargetSet_new(:, :, i) = muatargetGrid;
    mustargetSet_new(:, :, i) = mustargetGrid;

    fprintf('Iteration %d completed (parallel).\n', i);
end
%%
elapsedTime = toc(startTime);
hours = floor(elapsedTime / 3600); minutes = floor(mod(elapsedTime, 3600) / 60); seconds = mod(elapsedTime, 60);
%% 
% combining old and new data
muareconSet = cat(3, muareconSet, muareconSet_new);
muspreconSet = cat(3, muspreconSet, muspreconSet_new);
muatargetSet = cat(3, muatargetSet, muatargetSet_new);
mustargetSet = cat(3, mustargetSet, mustargetSet_new);

save(filename_mua_recon, 'muareconSet');
save(filename_mus_recon, 'muspreconSet');
save(filename_mua_target, 'muatargetSet');
save(filename_mus_target, 'mustargetSet');

%% saving to files containing training data 

filename_mua_recon = 'mua_recon.mat';
filename_mus_recon = 'mus_recon.mat';
filename_mua_target = 'mua_target.mat';
filename_mus_target = 'mus_target.mat';

% loading existing files, initializing files in the case they don't exist 
if isfile(filename_mua_recon)
    load(filename_mua_recon, 'muareconSet');
else
    muareconSet = [];
end

if isfile(filename_mus_recon)
    load(filename_mus_recon, 'muspreconSet');
else
    muspreconSet = [];
end

if isfile(filename_mua_target)
    load(filename_mua_target, 'muatargetSet');
else
    muatargetSet = [];
end

if isfile(filename_mus_target)
    load(filename_mus_target, 'mustargetSet');
else
    mustargetSet = [];
end

% appending new data
muareconSet = cat(3, muareconSet,   muareconSet_new);
muspreconSet = cat(3, muspreconSet,  muspreconSet_new);
muatargetSet = cat(3, muatargetSet,  muatargetSet_new);
mustargetSet = cat(3, mustargetSet,  mustargetSet_new);

% saving all data to the actual file
save(filename_mua_recon, 'muareconSet');
save(filename_mus_recon, 'muspreconSet');
save(filename_mua_target, 'muatargetSet');
save(filename_mus_target, 'mustargetSet');

% displaying how many reconstructions and targets (all should have the same amount) 
fprintf('Saved data:\n');
fprintf('  %s: %d reconstructions\n', filename_mua_recon,  size(muareconSet, 3));
fprintf('  %s: %d reconstructions\n', filename_mus_recon,  size(muspreconSet, 3));
fprintf('  %s: %d target images\n', filename_mua_target, size(muatargetSet, 3));
fprintf('  %s: %d target images\n', filename_mus_target, size(mustargetSet, 3));
