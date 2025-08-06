% uses diff_img_creating_validation.m, replaces for with parfor -> faster 

close all; clear all;
%% 

N = 16; % number of sources and detectors (16s16d)
numOfData = 100; % number of iteration in the loop
res = 32; % resolution of grid images 

% filenames 
filename_mua_recon = 'mua_valid_recon.mat';
filename_mus_recon = 'mus_valid_recon.mat';
filename_mua_target = 'mua_valid_target.mat';
filename_mus_target = 'mus_valid_target.mat';

% source positions 
ths = (0 : 2*pi/N : 2 * pi)';
sp = [25*cos(ths) 25*sin(ths)];

% detector positions 
thd = (pi/N : 2*pi/N : 2 * pi)';
mp = [25*cos(thd) 25*sin(thd)];

% loading mesh, setting sources and detectors to mesh 
meshname = 'mesh2D_r28.msh';
meshpath = 'meshfiles/';
hMesh = toastMesh([meshpath meshname]);
hMesh.SetQM(sp, mp);
hBasis = toastBasis(hMesh, [res res], 'Linear');

% FEM setup
qvec = hMesh.Qvec('Neumann', 'Gaussian', 2);
mvec = hMesh.Mvec('Gaussian', 2, 1.4);
ref = ones(hMesh.NodeCount(), 1) * 1.4;

% loading existing files
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

% temporary holdings for created data 
muareconSet_new = zeros(res, res, numOfData);
muspreconSet_new = zeros(res, res, numOfData);
muatargetSet_new = zeros(res, res, numOfData);
mustargetSet_new = zeros(res, res, numOfData);

%% loop 
rng('shuffle');

% change in angle between iterations, all inclusions have the same angle step to help prevent overlapping 
angle_step = deg2rad(55); 

% starting positions of inclusions 
start_angle1 = 0;
start_angle2 = 3 * pi/4;
start_angle3 = pi/4;
start_angle4 = 3*pi/2; 

numStepsPerCircle = 2 * pi / angle_step;
R_values = zeros(1, numOfData); % inclusions rotate about a "circle" with a radius R_values(i) 
rotCounters = zeros(1, numOfData);

if isempty(gcp('nocreate'))
    parpool;
end

for i = 1:numOfData
    rotCounters(i) = floor((i - 1) / round(numStepsPerCircle)); % number of full rotations  

    % random radius per rotation group
    rng(rotCounters(i) + 1); % seed with rotCounter + 1 
    R_values(i) = randi([10 18]); % larger radius to prevent overlapping 
end

parfor i = 1:numOfData  
    R = R_values(i);
    rotCounter = rotCounters(i);

    % local variables for each worker 
    hMesh_local = toastMesh([meshpath meshname]);
    hMesh_local.SetQM(sp, mp);
    hBasis_local = toastBasis(hMesh_local, [res res], 'Linear');

    % loading mesh data 
    [vtx, elem, ~] = hMesh_local.Data;
    n = size(vtx, 1);

    % inclusion 1 
    angle1 = mod(start_angle1 + (i - 1) * angle_step, 2*pi); % starting angle for inclusion 1
    r1 = randi([3 5]); % radius for inclusion 1
    cx1 = R * cos(angle1); % x-coordinate 
    cy1 = R * sin(angle1); % y-coordinate
    Index1 = find(sqrt((cx1 - vtx(:,1)).^2 + (cy1 - vtx(:,2)).^2) < r1); % index values that fall within radius r1

    % inclusion 2 
    angle2 = mod(start_angle2 + (i - 1) * angle_step, 2*pi);
    r2 = randi([3 5]); 
    cx2 = R * cos(angle2);
    cy2 = R * sin(angle2);
    Index2 = find(sqrt((cx2 - vtx(:,1)).^2 + (cy2 - vtx(:,2)).^2) < r2);
    
    % background values 
    mua0 = 0.01; 
    mus0 = 1;
    
    mus = mus0 * ones(n,1);
    mua = mua0 * ones(n,1);

    % adding inclusions with random coefficients
    % indexes in mus are scattering inclusions, in mua absorbing inclusions
    mus(Index1) = 0.5 + (1.5 - 0.5) * rand(); % [0.5, 1.5]
    mua(Index2) = 0.005 + (0.02 - 0.005) * rand(); % [0.005, 0.02]

    % inclusion 3 
    if (randi[0 1])
        angle3 = mod(start_angle3 + (i - 1) * angle_step, 2*pi);
        r3 = randi([3 5]);
        cx3 = R * cos(angle3);
        cy3 = R * sin(angle3);
        Index3 = find(sqrt((cx3 - vtx(:,1)).^2 + (cy3 - vtx(:,2)).^2) < r3);
        mua(Index3) = 0.005 + (0.02 - 0.005) * rand(); % [0.005, 0.02]
    end 
    
    % inclusion 4 
    if (randi[0 1])
        angle4 = mod(start_angle4 + (i - 1) * angle_step, 2*pi);
        r4 = randi([3 5]);
        cx4 = R * cos(angle4);
        cy4 = R * sin(angle4);
        Index4 = find(sqrt((cx4 - vtx(:,1)).^2 + (cy4 - vtx(:,2)).^2) < r4);
        mus(Index4) = 0.5 + (1.5 - 0.5) * rand(); % [0.5, 1.5]
    end
    
    % difference from background values 
    dmuatgt = mua - mua0;
    dmustgt = mus - mus0;

    % measurements with inclusions 
    freq = 100; % MHz
    K = dotSysmat(hMesh_local, mua, mus, ref, freq);
    phi = K \ qvec;
    gamma = mvec.' * phi;
    y = [real(log(gamma(:))); imag(log(gamma(:)))];

    % reference values without inclusions 
    mus_ref = mus0 * ones(n,1);
    mua_ref = mua0 * ones(n,1);
    K0 = dotSysmat(hMesh_local, mua_ref, mus_ref, ref, freq);
    phi0 = K0 \ qvec;
    gamma0 = mvec.' * phi0;
    y0 = [real(log(gamma0(:))); imag(log(gamma0(:)))];

    % difference in ... 
    deltay = y - y0;

    % random percentage of Gaussian noise 
    noise = 0.005 + (0.015 - 0.005) * rand(); % [0.005, 0.015]
    deltay = deltay + noise * randn(length(deltay), 1) .* abs(deltay);

    % Jacobian 
    J = toastJacobian(hMesh_local, [], qvec, mvec, mua_ref, mus_ref, ref, freq);
    stdn = noise * abs(y);
    Le = diag(1 ./ stdn);

    % prior 
    r_prior = 8;
    prior_std_mua = mua0 / 10;
    prior_std_mus = mus0 / 10;
    Lxmua = PriorOrnsteinUhlenbeck(struct('g', vtx), prior_std_mua, r_prior);
    Lxmus = PriorOrnsteinUhlenbeck(struct('g', vtx), prior_std_mus, r_prior);
    Lx = [Lxmua sparse(n,n); sparse(n,n) Lxmus];

    % solving the system 
    x = ([Le * J; Lx]) \ ([Le * deltay; zeros(2 * n, 1)]);
    muarecon = x(1:n);
    musprecon = x(n+1:end);

    % transfering images from mesh to grid and rotating 
    muareconGrid = rot90(reshape(hBasis_local.Map('M->B', muarecon), [res, res]), 1);
    muspreconGrid = rot90(reshape(hBasis_local.Map('M->B', musprecon), [res, res]), 1);
    muatargetGrid = rot90(reshape(hBasis_local.Map('M->B', dmuatgt), [res, res]), 1);
    mustargetGrid = rot90(reshape(hBasis_local.Map('M->B', dmustgt), [res, res]), 1);

    % appending grid image to temporary holding
    muareconSet_new(:, :, i) = muareconGrid;
    muspreconSet_new(:, :, i) = muspreconGrid;
    muatargetSet_new(:, :, i) = muatargetGrid;
    mustargetSet_new(:, :, i) = mustargetGrid;

    fprintf('Iteration %d completed.\n', i);
end

% append and save created data
muareconSet = cat(3, muareconSet, muareconSet_new);
muspreconSet = cat(3, muspreconSet, muspreconSet_new);
muatargetSet = cat(3, muatargetSet, muatargetSet_new);
mustargetSet = cat(3, mustargetSet, mustargetSet_new);

save(filename_mua_recon, 'muareconSet');
save(filename_mus_recon, 'muspreconSet');
save(filename_mua_target, 'muatargetSet');
save(filename_mus_target, 'mustargetSet');

fprintf('Saved data:\n');
fprintf('  %s: %d reconstructions\n', filename_mua_recon,  size(muareconSet, 3));
fprintf('  %s: %d reconstructions\n', filename_mus_recon,  size(muspreconSet, 3));
fprintf('  %s: %d target images\n', filename_mua_target, size(muatargetSet, 3));
fprintf('  %s: %d target images\n', filename_mus_target, size(mustargetSet, 3));
