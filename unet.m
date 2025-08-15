% In this script, U-Net neural network is trained using training data and validation data created in diff_img_creating_data_v3.m
% Variable "percentage" can be changed.
% Training options can also be altered. 

% Sources: 
% https://github.com/prafful-kumar/Blurred-Image-Recognition/blob/main/Deblurring_U_NET(Keras).ipynb
% https://se.mathworks.com/help/deeplearning/builtin-layers.html
% https://se.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.dropoutlayer.html

clear; close all;

%% Loading training data and validation data created using diff_img_creating_data_v2.m (or older versions) 

load("mua_recon.mat"); load("mua_target.mat")                % files contain µa reconstructions and targets used for training 
muareconMatrix = muareconSet;                    
muatargetMatrix = muatargetSet;

load("mus_recon.mat"); load("mus_target.mat")                % files contain µs' reconstructions and targets used for training 
muspreconMatrix = muspreconSet;
mustargetMatrix = mustargetSet;

load("mua_valid_recon.mat"); load("mua_valid_target.mat")    % files contain µa reconstructions and targets used for validation
muaValReconSet = muareconSet; 
muaValTargetSet = muatargetSet;

load("mus_valid_recon.mat"); load("mus_valid_target.mat")    % files contain µs' reconstructions and targets used for validation 
musValReconSet = muspreconSet;
musValTargetSet = mustargetSet;

%% Assigning constants and scalining data 

percent = 0.25;                                              % percent of disabled neurons in the network (dropout)
res = 32;                                                    % resolution of training data, validation data
% numImages = size(muareconMatrix, 3);                       % number of dataset/images in files
                                                             % In file diff_img_creating_data_v2.m µa, µs' are constrained to a range of min, max values. This gives the ranges for both. 
mua_range = 0.015;                                           % µa is in [0.005, 0.02]  -> range = 0.02 - 0.005 = 0.015
mus_range = 1.5;                                               % µs' is in [0.5, 2]    -> range = 2 - 0.5 = 1.5

input_mua = muareconMatrix / mua_range;                      % Inputs and targets need to be scaled so that µa and µs' values are on the same range,
input_mus = muspreconMatrix / mus_range;                     % before training the network. After using the trained net to make a prediction, the outputs need to
                                                             % be scaled back. 
target_mua = muatargetMatrix / mua_range;
target_mus = mustargetMatrix / mus_range;

input_data = cat(4, input_mua, input_mus);                   % combining the inputs to a 2-channel format 
input_data = permute(input_data, [1 2 4 3]);                 % permuting [res × res × numImages × 2] -> [res × res × 2 × numImages]
                                                             
target_data = cat(4, target_mua, target_mus);                % combining targets to a 2-channel format 
target_data = permute(target_data, [1 2 4 3]);               % permuting [res × res × numImages × 2] -> [res × res × 2 × numImages]

input_mua_val = mua_val_recon_set / mua_range;               % scaling µa validation-reconstruction data 
input_mus_val = mus_val_recon_set / mus_range;               % scaling µs' validation-reconstruction data

target_mua_val = mua_val_target_set / mua_range;             % scaling µa validation-target data
target_mus_val = mus_val_target_set / mus_range;             % scaling µs' validation-target data 

input_val = cat(4, input_mua_val, input_mus_val);            % combining the inputs for µa, µs' into a 2-channel format
input_val = permute(input_val, [1 2 4 3]);                   % permuting 

target_val = cat(4, target_mua_val, target_mus_val);         % combining the targets for µa, µs' into a 2-channel format 
target_val = permute(target_val, [1 2 4 3]);                 % permuting 

%% creating datastores 

input_DS = arrayDatastore(input_data, 'IterationDimension', 4);   % datastore for input data, 'IterationDimension' is 4 referring to how input_data was permuted 
target_DS = arrayDatastore(target_data, 'IterationDimension', 4); % datastore for target data 
train_DS = combine(input_DS, target_DS);                          % combining the datastores, creating the set of training data

input_val_DS = arrayDatastore(input_val, 'IterationDimension', 4);   % datastore for input data, used for validation
target_val_DS = arrayDatastore(target_val, 'IterationDimension', 4); % datastore for target data, used for validation
val_DS = combine(input_val_DS, target_val_DS);                       % combining datastores, creating datastore used for validation 

%% Building the network 

% Input layer
inputLayer = imageInputLayer([res res 2], 'Name', 'input');

% Encoder
enc1 = [
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'enc1_conv1') % starts with 64 filters
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

% Bottleneck
bottleneck = [
    convolution2dLayer(3, 1024, 'Padding', 'same', 'Name', 'bottleneck_conv1')
    batchNormalizationLayer('Name', 'bottleneck_bn1')
    reluLayer('Name', 'bottleneck_relu1')
    
    convolution2dLayer(3, 1024, 'Padding', 'same', 'Name', 'bottleneck_conv2')
    batchNormalizationLayer('Name', 'bottleneck_bn2')
    reluLayer('Name', 'bottleneck_relu2')

    dropoutLayer(percent, 'Name', 'bottleneck_dropout') 

];

% Decoder
dec4 = [
    transposedConv2dLayer(2, 512, 'Stride', 2, 'Name', 'dec4_up')
    reluLayer('Name', 'dec4_relu1')
    depthConcatenationLayer(2, 'Name', 'dec4_concat')
    
    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'dec4_conv1')
    batchNormalizationLayer('Name', 'dec4_bn1')
    reluLayer('Name', 'dec4_relu2')

    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'dec4_conv2')
    batchNormalizationLayer('Name', 'dec4_bn2')
    reluLayer('Name', 'dec4_relu3')
];

dec3 = [
    transposedConv2dLayer(2, 256, 'Stride', 2, 'Name', 'dec3_up')
    reluLayer('Name', 'dec3_relu1')
    depthConcatenationLayer(2, 'Name', 'dec3_concat')
    
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'dec3_conv1')
    batchNormalizationLayer('Name', 'dec3_bn1')
    reluLayer('Name', 'dec3_relu2')

    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'dec3_conv2')
    batchNormalizationLayer('Name', 'dec3_bn2')
    reluLayer('Name', 'dec3_relu3')
];

dec2 = [
    transposedConv2dLayer(2, 128, 'Stride', 2, 'Name', 'dec2_up')
    reluLayer('Name', 'dec2_relu1')
    depthConcatenationLayer(2, 'Name', 'dec2_concat')
    
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'dec2_conv1')
    batchNormalizationLayer('Name', 'dec2_bn1')
    reluLayer('Name', 'dec2_relu2')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'dec2_conv2')
    batchNormalizationLayer('Name', 'dec2_bn2')
    reluLayer('Name', 'dec2_relu3')
];

dec1 = [
    transposedConv2dLayer(2, 64, 'Stride', 2, 'Name', 'dec1_up')
    reluLayer('Name', 'dec1_relu1')
    depthConcatenationLayer(2, 'Name', 'dec1_concat')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'dec1_conv1')
    batchNormalizationLayer('Name', 'dec1_bn1')
    reluLayer('Name', 'dec1_relu2')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'dec1_conv2')
    batchNormalizationLayer('Name', 'dec1_bn2')
    reluLayer('Name', 'dec1_relu3')
];

% Output
outputLayer = [
    convolution2dLayer(1, 2, 'Name', 'final_conv')  
    regressionLayer('Name', 'output') % loss function MSE 
];


% Assembling the layer graph
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

% Encoder connections
lgraph = connectLayers(lgraph, 'input', 'enc1_conv1');
lgraph = connectLayers(lgraph, 'enc1_pool', 'enc2_conv1');
lgraph = connectLayers(lgraph, 'enc2_pool', 'enc3_conv1');
lgraph = connectLayers(lgraph, 'enc3_pool', 'enc4_conv1');
lgraph = connectLayers(lgraph, 'enc4_pool', 'bottleneck_conv1');

% Decoder connections with skip connections
lgraph = connectLayers(lgraph, 'bottleneck_dropout', 'dec4_concat/in1');
lgraph = connectLayers(lgraph, 'enc4_relu2', 'dec4_concat/in2');
lgraph = connectLayers(lgraph, 'dec4_relu3', 'dec3_concat/in1');
lgraph = connectLayers(lgraph, 'enc3_relu2', 'dec3_concat/in2');
lgraph = connectLayers(lgraph, 'dec3_relu3', 'dec2_concat/in1');
lgraph = connectLayers(lgraph, 'enc2_relu2', 'dec2_concat/in2');
lgraph = connectLayers(lgraph, 'dec2_relu3', 'dec1_concat/in1');
lgraph = connectLayers(lgraph, 'enc1_relu2', 'dec1_concat/in2');

% Output connection
lgraph = connectLayers(lgraph, 'dec1_relu3', 'final_conv');

% analyzeNetwork(lgraph) % shows networks structure 

%% Specifying the training options and training the net 

options = trainingOptions('adam', ...
    'MaxEpochs', 125, ...
    'InitialLearnRate', 1e-4, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'ValidationData', val_DS, ...
    'ValidationFrequency', 50, ...
    'ExecutionEnvironment', 'gpu');

startTime = tic;
[net, info] = trainNetwork(train_DS, lgraph, options);
elapsedTime = toc(startTime); 

% Saving trained net, info and elapsedTime 
save('unet.mat', 'net', 'info', 'elapsedTime');

% Displaying a confirmation message
disp('Done.');


