clear; close all;

load("360_mua_recon.mat"); load("360_mua_target.mat")
muareconMatrix = muareconSet;
muatargetMatrix = muatargetSet;

load("360_mus_recon.mat"); load("360_mus_target.mat")
muspreconMatrix = muspreconSet;
mustargetMatrix = mustargetSet;

load("360_mua_valid_recon.mat"); load("360_mua_valid_target.mat")
muaValReconSet = muareconSet; 
muaValiTargetSet = muatargetSet;

load("360_mus_valid_recon.mat"); load("360_mus_valid_target.mat")
musValReconSet = muspreconSet;
musValTargetSet = mustargetSet;

res = 32;
%% 
% number of images in the input data set(s)
numImages = size(muareconMatrix, 3); 

% background values
mua_bg = 0.01;
mus_bg = 1.0;

% max absolute deviation for scaling
mua_range = 0.01; % mua in [0.005, 0.02]
mus_range = 0.5; % mus in [0.5, 1.5] 

% scaling values for network training 
input_mua = (muareconMatrix - mua_bg) / mua_range; % scaled
input_mus = (muspreconMatrix - mus_bg) / mus_range; % scaled

target_mua = (muatargetMatrix - mua_bg) / mua_range; % scaled
target_mus = (mustargetMatrix - mus_bg) / mus_range; % scaled

% combining inputs and targets into 2-channel format
inputData = cat(4, input_mua, input_mus);  % [res × res × numImages × 2]
inputData = permute(inputData, [1 2 4 3]); % -> [res × res × 2 × numImages]

targetData = cat(4, target_mua, target_mus);  
targetData = permute(targetData, [1 2 4 3]); 

% validation data
input_mua_val = (muaValReconSet - mua_bg) / mua_range; 
input_mus_val = (musValReconSet - mus_bg) / mus_range; 

target_mua_val = (muaValTargetSet - mua_bg) / mua_range; % [0, 1]
target_mus_val = (musValTargetSet - mus_bg) / mus_range; % [0, 1]

inputVal = cat(4, input_mua_val, input_mus_val);
inputVal = permute(inputVal, [1 2 4 3]); 

targetVal = cat(4, target_mua_val, target_mus_val);
targetVal = permute(targetVal, [1 2 4 3]);

% source: 
% https://stackoverflow.com/questions/52527210/how-to-make-training-data-as-a-4-d-array-in-neural-network-matlab-proper-way-t

%% combining datastores 

% input data and targets 
inputDS = arrayDatastore(inputData, 'IterationDimension', 4);
targetDS = arrayDatastore(targetData, 'IterationDimension', 4);
dsTrain = combine(inputDS, targetDS);

% validation data and targets 
inputValDS = arrayDatastore(inputVal, 'IterationDimension', 4);
targetValDS = arrayDatastore(targetVal, 'IterationDimension', 4);
valDS = combine(inputValDS, targetValDS);

%% building the network 

% built-in layers (https://se.mathworks.com/help/deeplearning/builtin-layers.html)
% compare to this: https://github.com/prafful-kumar/Blurred-Image-Recognition/blob/main/Deblurring_U_NET(Keras).ipynb

% input layer
inputLayer = imageInputLayer([32 32 2], 'Name', 'input');

% encoder
enc1 = [
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'enc1_conv1') % starts with 64 filters instead of 32 
    batchNormalizationLayer('Name', 'enc1_bn1')
    reluLayer('Name', 'enc1_relu1')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'enc1_conv2')
    batchNormalizationLayer('Name', 'enc1_bn2')
    reluLayer('Name', 'enc1_relu2')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc1_pool')
];

enc2 = [
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'enc2_conv1')
    batchNormalizationLayer('Name', 'enc2_bn1')
    reluLayer('Name', 'enc2_relu1')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'enc2_conv2')
    batchNormalizationLayer('Name', 'enc2_bn2')
    reluLayer('Name', 'enc2_relu2')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc2_pool')
];

enc3 = [
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'enc3_conv1')
    batchNormalizationLayer('Name', 'enc3_bn1')
    reluLayer('Name', 'enc3_relu1')

    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'enc3_conv2')
    batchNormalizationLayer('Name', 'enc3_bn2')
    reluLayer('Name', 'enc3_relu2')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc3_pool')
];

enc4 = [
    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'enc4_conv1')
    batchNormalizationLayer('Name', 'enc4_bn1')
    reluLayer('Name', 'enc4_relu1')

    convolution2dLayer(3, 512, 'Padding','same', 'Name', 'enc4_conv2')
    batchNormalizationLayer('Name', 'enc4_bn2')
    reluLayer('Name', 'enc4_relu2')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc4_pool')
];

% bottleneck
bottleneck = [
    convolution2dLayer(3, 1024, 'Padding', 'same', 'Name', 'bottleneck_conv1')
    batchNormalizationLayer('Name', 'bottleneck_bn1')
    reluLayer('Name', 'bottleneck_relu1')
    
    convolution2dLayer(3, 1024, 'Padding', 'same', 'Name', 'bottleneck_conv2')
    batchNormalizationLayer('Name', 'bottleneck_bn2')
    reluLayer('Name', 'bottleneck_relu2')

    dropoutLayer(0.25, 'Name', 'bottleneck_dropout')  % diables 25% of neurons randomly to prevent overfitting -> net doesn't depend on specific neurons 
    % source: https://www.mdpi.com/2079-9292/11/3/305#:~:text=Following%20this%20process%2C%20the%20feature%20maps%20enter%20a%20batch%20normalization%20and%20ReLU%20activation%20layer.%20After%20which%2C%20they%20pass%20through%20a%20dropout%20layer%20(25%25%20dropout).
    
    transposedConv2dLayer(2, 512, 'Stride', 2, 'Name', 'bottleneck_up')
    reluLayer('Name', 'bottleneck_relu3')
];

% decoder
dec4 = [
    depthConcatenationLayer(2, 'Name', 'dec4_concat')
    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'dec4_conv1')
    batchNormalizationLayer('Name', 'dec4_bn1')
    reluLayer('Name', 'dec4_relu1')

    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'dec4_conv2')
    batchNormalizationLayer('Name', 'dec4_bn2')
    reluLayer('Name', 'dec4_relu2')

    transposedConv2dLayer(2, 256, 'Stride', 2, 'Name', 'dec4_up')
    reluLayer('Name', 'dec4_relu3')
];

dec3 = [
    depthConcatenationLayer(2, 'Name', 'dec3_concat')
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'dec3_conv1')
    batchNormalizationLayer('Name', 'dec3_bn1')
    reluLayer('Name', 'dec3_relu1')

    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'dec3_conv2')
    batchNormalizationLayer('Name', 'dec3_bn2')
    reluLayer('Name', 'dec3_relu2')

    transposedConv2dLayer(2, 128, 'Stride', 2, 'Name', 'dec3_up')
    reluLayer('Name', 'dec3_relu3')
];

dec2 = [
    depthConcatenationLayer(2, 'Name', 'dec2_concat')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'dec2_conv1')
    batchNormalizationLayer('Name', 'dec2_bn1')
    reluLayer('Name', 'dec2_relu1')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'dec2_conv2')
    batchNormalizationLayer('Name', 'dec2_bn2')
    reluLayer('Name', 'dec2_relu2')

    transposedConv2dLayer(2, 64, 'Stride', 2, 'Name', 'dec2_up')
    reluLayer('Name', 'dec2_relu3')
];

dec1 = [
    depthConcatenationLayer(2, 'Name', 'dec1_concat')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'dec1_conv1')
    batchNormalizationLayer('Name', 'dec1_bn1')
    reluLayer('Name', 'dec1_relu1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'dec1_conv2')
    batchNormalizationLayer('Name', 'dec1_bn2')
    reluLayer('Name', 'dec1_relu2')
];

% output
outputLayer = [
    convolution2dLayer(1, 2, 'Name', 'final_conv')  
    regressionLayer('Name', 'output') % makes the loss function to mean squared error 
];


% assembling
lgraph = layerGraph();
lgraph = addLayers(lgraph, inputLayer);
lgraph = addLayers(lgraph, enc1);
lgraph = addLayers(lgraph, enc2);
lgraph = addLayers(lgraph, enc3);
lgraph = addLayers(lgraph, enc4);
lgraph = addLayers(lgraph, bottleneck);
lgraph = addLayers(lgraph, dec4);
lgraph = addLayers(lgraph, dec3);
lgraph = addLayers(lgraph, dec2);
lgraph = addLayers(lgraph, dec1);
lgraph = addLayers(lgraph, outputLayer);

% encoder connections
lgraph = connectLayers(lgraph, 'input', 'enc1_conv1');
lgraph = connectLayers(lgraph, 'enc1_pool', 'enc2_conv1');
lgraph = connectLayers(lgraph, 'enc2_pool', 'enc3_conv1');
lgraph = connectLayers(lgraph, 'enc3_pool', 'enc4_conv1');
lgraph = connectLayers(lgraph, 'enc4_pool', 'bottleneck_conv1');

% decoder connections with skip connections
lgraph = connectLayers(lgraph, 'bottleneck_relu3', 'dec4_concat/in1');
lgraph = connectLayers(lgraph, 'enc4_relu2', 'dec4_concat/in2');
lgraph = connectLayers(lgraph, 'dec4_relu3', 'dec3_concat/in1');
lgraph = connectLayers(lgraph, 'enc3_relu2', 'dec3_concat/in2');
lgraph = connectLayers(lgraph, 'dec3_relu3', 'dec2_concat/in1');
lgraph = connectLayers(lgraph, 'enc2_relu2', 'dec2_concat/in2');
lgraph = connectLayers(lgraph, 'dec2_relu3', 'dec1_concat/in1');
lgraph = connectLayers(lgraph, 'enc1_relu2', 'dec1_concat/in2');

% output connection
lgraph = connectLayers(lgraph, 'dec1_relu2', 'final_conv');

% analyzeNetwork(lgraph) % shows networks structure 

%% specifying the training options and training the net 
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'InitialLearnRate', 1e-4, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'ValidationData', valDS, ...
    'ValidationFrequency', 50, ...
    'ExecutionEnvironment', 'gpu'); % is GPU is not available use 'parallel' or 'cpu' 

startTime = tic;
% training the net accoring to the options 
[net, info] = trainNetwork(trainDS, lgraph, options);
elapsedTime = toc(startTime); 

% saving trained net, info and elapsedTime 
save('unet.mat', 'net', 'info', 'elapsedTime');

% displaying a confirmation message
disp('Done.');
