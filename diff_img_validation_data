close; clear all;

filename_mua_recon = '360_mua_valid_recon.mat';
filename_mus_recon = '360_mus_valid_recon.mat';
filename_mua_target = '360_mua_valid_target.mat';
filename_mus_target = '360_mus_valid_target.mat';

N = 16; % number of sources (and number of detectors) in range [0, 2pi]
numOfData = 100; % number of reconstructions 
res = 32; % resolution

ths = (0 : 2*pi/N : 2 * pi)';
% ths = (pi/2 : 2*pi/N : 3*pi/2)'; 
sp = [25*cos(ths) 25*sin(ths)];

thd = (pi/N : 2*pi/N : 2 * pi)';
% thd = (pi/N + pi/2 : 2*pi/N : 3*pi/2)';
mp = [25*cos(thd) 25*sin(thd)];

meshname = 'mesh2D_r28.msh';
meshpath = 'meshfiles/';
hMesh = toastMesh([meshpath meshname]);
hMesh.SetQM(sp, mp);

% Basis for mapping reconstructions to 2D grids
hBasis = toastBasis(hMesh, [res res], 'Linear');

% FEM setup
qvec = hMesh.Qvec('Neumann', 'Gaussian', 2);
mvec = hMesh.Mvec('Gaussian', 2, 1.4);
ref = ones(hMesh.NodeCount(), 1) * 1.4;

% check and load existing files if they exist
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

% for validation data inclusions are created so that they do not overlap -> polar coordinates 

angle_step = deg2rad(5); % same angle step for all inclusions 

% starting angle for each inclusion 
start_angle1 = pi;
start_angle2 = 3 * pi/2;
start_angle3 = pi/2;
start_angle4 = 0; 

% inclusions rotate with a radius R which changes after each full rotation 
R = randi([4 12]);
numStepsPerCircle = 2 * pi / angle_step;
rotCounter = 0;

for i = 1:numOfData  
    % every full rotation triggers a new radius 
    if mod(i, round(numStepsPerCircle)) == 1 && i > 1
            rotCounter = rotCounter + 1; 
            R = randi([4 12]); 
    end

    % Load mesh data
    [vtx, elem, ~] = hMesh.Data;
    n = size(vtx, 1);

    % inclusion 1 (scattering)
    angle1 = mod(start_angle1 + (i - 1) * angle_step, 2*pi);
    r1 = randi([3 5]); 
    cx1 = R * cos(angle1);
    cy1 = R * sin(angle1);
    Index1 = find(sqrt((cx1 - vtx(:,1)).^2 + (cy1 - vtx(:,2)).^2) < r1);

    % inclusion 2 (absorption)
    angle2 = mod(start_angle2 + (i - 1) * angle_step, 2*pi);
    r2 = randi([3 5]); 
    cx2 = R * cos(angle2);
    cy2 = R * sin(angle2);
    Index2 = find(sqrt((cx2 - vtx(:,1)).^2 + (cy2 - vtx(:,2)).^2) < r2);

    % inclusion 3 (another absorption)
    % r3 = randi([3 5]);
    % cx3 = R * cos(angle3);
    % cy3 = R * sin(angle3);
    % Index3 = find(sqrt((cx3 - vtx(:,1)).^2 + (cy3 - vtx(:,2)).^2) < r3);
    
    % inclusion 4 (another scattering)
    % r4 = randi([3 5]);
    % cx4 = R * cos(angle4);
    % cy4 = R * sin(angle4);
    % Index4 = find(sqrt((cx4 - vtx(:,1)).^2 + (cy4 - vtx(:,2)).^2) < r4);

    % background values 
    mua0 = 0.01; 
    mus0 = 1;

    mus = mus0 * ones(n,1);
    mua = mua0 * ones(n,1);
    
    mus(Index1) = 0.5 + (1.5 - 0.5) * rand(); % [0.5, 1.5]
    % mus(Index4) = 0.5 + (1.5 - 0.5) * rand(); % [0.5, 1.5]

    mua(Index2) = 0.005 + (0.02 - 0.005) * rand(); % [0.005, 0.05], -> [0.005, 0.02]
    % mua(Index3) = 0.005 + (0.02 - 0.005) * rand(); % [0.005, 0.05]

    dmuatgt = mua - mua0;
    dmustgt = mus - mus0;

    % Generate measurement with inclusions
    freq = 100; % MHz
    K = dotSysmat(hMesh, mua, mus, ref, freq);
    phi = K \ qvec;
    gamma = mvec.' * phi;
    y = [real(log(gamma(:))); imag(log(gamma(:)))];

    % Reference measurement (no inclusions)
    mus_ref = mus0 * ones(n,1);
    mua_ref = mua0 * ones(n,1);
    K0 = dotSysmat(hMesh, mua_ref, mus_ref, ref, freq);
    phi0 = K0 \ qvec;
    gamma0 = mvec.' * phi0;
    y0 = [real(log(gamma0(:))); imag(log(gamma0(:)))];

    deltay = y - y0;

    % Gaussian noise
    noise = 0.005 + (0.015 - 0.005) * rand(); % [0.005, 0.015]
    deltay = deltay + noise * randn(length(deltay), 1) .* abs(deltay);

    % Build Jacobian and prior
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

    % Map to 2D image (grid format)
    muareconGrid = rot90(reshape(hBasis.Map('M->B', muarecon), [res, res]), 1);
    muspreconGrid = rot90(reshape(hBasis.Map('M->B', musprecon), [res, res]), 1);
    muatargetGrid = rot90(reshape(hBasis.Map('M->B', dmuatgt), [res, res]), 1);
    mustargetGrid = rot90(reshape(hBasis.Map('M->B', dmustgt), [res, res]), 1);

    muareconSet(:, :, nExistingMua + i) = muareconGrid;
    muspreconSet(:, :, nExistingMus + i) = muspreconGrid;
    muatargetSet(:, :, nExistingMuaTarget + i) = muatargetGrid;
    mustargetSet(:, :, nExistingMusTarget + i) = mustargetGrid;

    fprintf('Iteration %d completed.\n', i);
end

% Save all reconstructions and targets
save(filename_mua_recon, 'muareconSet');
save(filename_mus_recon, 'muspreconSet');
save(filename_mua_target, 'muatargetSet');
save(filename_mus_target, 'mustargetSet');

numImages = size(muareconSet, 3);
fprintf("%d reconstructions in file.\n", numImages);
