clear; close all;

startTime = tic();

res = 32;
numOfData = 1000;
rng('shuffle');

addpath('./functions/');
dataDir = './data/';
meshDir = './meshfiles/';

load([dataDir 'measConfig.mat']);
load([dataDir 'stdError.mat']);

Geometry = createGeometry(meshName = "mesh2D_r28.msh", meshDir = meshDir, isGmsh = 0, ...
    measConfig = measConfig, dataDir = dataDir, gridDimensions = [res res]);

hMesh = Geometry.hMesh;
hBasis = Geometry.hBasis;
qvec = Geometry.qVec;
mvec = Geometry.mVec;  
ref = Geometry.refIndVec;
DataLinkList = hMesh.DataLinkList();

[vtx, elem, ~] = hMesh.Data(); % 80mm x 80mm 
n = size(vtx, 1);

muareconSet_new = zeros(res, res, numOfData);
muspreconSet_new = zeros(res, res, numOfData);
muatargetSet_new = zeros(res, res, numOfData);
mustargetSet_new = zeros(res, res, numOfData);

mua0 = 0.01; mus0 = 1;

min_mua = 0.005; max_mua = 0.02;
min_mus = 0.5; max_mus = 1.5;

min_noise = 0.005; max_noise = 0.03; 

min_incl_radius = 3; max_incl_radius = 12; 

min_x = -35; max_x = 35;
min_y = -35; max_y = 35;

fprintf('Progress: %3d%% (%3d / %3d)', 0, 0, numOfData);
backspace_count = length(sprintf('%3d%% (%3d / %3d)', 0, 0, numOfData));

for i = 1:numOfData
    dmuatgt = 0;
    dmustgt = 0;

    % while-loop is repeated until difference between background and target are not zero 
    while all(dmuatgt == 0) || all(dmustgt == 0)
    r1 = randi([min_incl_radius max_incl_radius]);
    cx1 = min_x + (max_x - min_x) * rand();
    cy1 = min_y + (max_y - min_y) * rand();
    Index1 = find(sqrt((cx1 - vtx(:,1)).^2 + (cy1 - vtx(:,2)).^2) < r1);
    
    r2 = randi([min_incl_radius max_incl_radius]);
    cx2 = min_x + (max_x - min_x) * rand();
    cy2 = min_y + (max_y - min_y) * rand();
    Index2 = find(sqrt((cx2 - vtx(:,1)).^2 + (cy2 - vtx(:,2)).^2) < r2);

    mus = mus0 * ones(n,1);
    mua = mua0 * ones(n,1);

    mus(Index1) = min_mus + (max_mus - min_mus) * rand();
    mua(Index2) = min_mua + (max_mua - min_mua) * rand();

    if (randi([0 1]))
        r3 = randi([min_incl_radius max_incl_radius]);
        cx3 = min_x + (max_x - min_x) * rand();
        cy3 = min_x + (max_x - min_x) * rand();
        Index3 = find(sqrt((cx3 - vtx(:,1)).^2 + (cy3 - vtx(:,2)).^2) < r3);
        mua(Index3) = min_mua + (max_mua - min_mua) * rand();
    end

    if (randi([0 1]))
        r4 = randi([min_incl_radius max_incl_radius]);
        cx4 = min_x + (max_x - min_x) * rand();
        cy4 = min_x + (max_x - min_x) * rand();
        Index4 = find(sqrt((cx4 - vtx(:,1)).^2 + (cy4 - vtx(:,2)).^2) < r4);
        mus(Index4) = min_mus + (max_mus - min_mus) * rand();
    end

    dmuatgt = mua - mua0;
    dmustgt = mus - mus0;
    end 
    freq = 100;

    % K = dotSysmat(hMesh, mua, mus, ref, freq);
    % phi = K \ qvec;
    % gamma = mvec.' * phi;
    % y = [real(log(gamma(:))); imag(log(gamma(:)))];

    K = dotSysmat(hMesh, mua, mus, ref, freq);
    phi = K \ qvec;
    gamma = mvec.' * phi;
    gamma = gamma(DataLinkList());
    log_gamma = log(gamma(:));
    real_part = real(log_gamma);
    imag_part = unwrap(imag(log_gamma));
    
    y = [real_part; imag_part];
    
    % mus_ref = mus0 * ones(n,1);
    % mua_ref = mua0 * ones(n,1);
    % K0 = dotSysmat(hMesh, mua_ref, mus_ref, ref, freq);
    % phi0 = K0 \ qvec;
    % gamma0 = mvec.' * phi0;
    % y0 = [real(log(gamma0(:))); imag(log(gamma0(:)))];

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
   
    deltay = y - y0;

    noise = min_noise + (max_noise - min_noise) * rand();
    deltay = deltay + noise * randn(length(deltay), 1) .* abs(deltay);

    J = toastJacobian(hMesh, [], qvec, mvec, mua_ref, mus_ref, ref, freq);
    % stdn = noise * abs(y);
    stdn = stdError;
    Le = diag(1 ./ stdn);

    r_prior = 8;
    prior_std_mua = mua0;
    prior_std_mus = mus0;
    Lxmua = PriorOrnsteinUhlenbeck(struct('g', vtx), prior_std_mua, r_prior);
    Lxmus = PriorOrnsteinUhlenbeck(struct('g', vtx), prior_std_mus, r_prior);
    Lx = [Lxmua sparse(n,n); sparse(n,n) Lxmus];

    x = ([Le * J; Lx]) \ ([Le * deltay; zeros(2 * n, 1)]);
    muarecon = x(1:n);
    musprecon = x(n+1:end);

    muareconGrid = rot90(reshape(hBasis.Map('M->B', muarecon), [res, res]), 1);
    muspreconGrid = rot90(reshape(hBasis.Map('M->B', musprecon), [res, res]), 1);
    muatargetGrid = rot90(reshape(hBasis.Map('M->B', dmuatgt), [res, res]), 1);
    mustargetGrid = rot90(reshape(hBasis.Map('M->B', dmustgt), [res, res]), 1);

    muareconSet_new(:, :, i) = muareconGrid;
    muspreconSet_new(:, :, i) = muspreconGrid;
    muatargetSet_new(:, :, i) = muatargetGrid;
    mustargetSet_new(:, :, i) = mustargetGrid;

    progress_str = sprintf(' %3d%% (%3d / %3d)', round(100*i/numOfData), i, numOfData);
    fprintf(['\b' repmat('\b', 1, backspace_count) '%s'], progress_str);
end
    
%% saving to a file 

filename_recon_mua = 'mua_recon.mat';
filename_recon_mus = 'mus_recon.mat';
filename_target_mua = 'mua_target.mat';
filename_target_mus = 'mus_target.mat';

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


muareconSet = cat(3, muareconSet, muareconSet_new);
muspreconSet = cat(3, muspreconSet, muspreconSet_new);
muatargetSet = cat(3, muatargetSet, muatargetSet_new);
mustargetSet = cat(3, mustargetSet, mustargetSet_new);

save(filename_recon_mua, 'muareconSet');
save(filename_recon_mus, 'muspreconSet');
save(filename_target_mua, 'muatargetSet');
save(filename_target_mus, 'mustargetSet');

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
