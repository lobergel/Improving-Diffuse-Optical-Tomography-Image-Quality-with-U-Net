clear; close all;

load("diff_MuaReconstructedData.mat"); load("diff_MuaTargetData.mat")
load("diff_MusReconstructedData.mat"); load("diff_MusTargetData.mat")
% inputData = musGridMatrix; 

res = 32;
%% 

inputDataScaled = (muareconMatrix - min(muareconMatrix(:))) / (max(muareconMatrix(:)) - min(muareconMatrix(:)));

% normalizing input and target matrices
input_mua = (muareconMatrix - min(muareconMatrix(:))) / ...
            (max(muareconMatrix(:)) - min(muareconMatrix(:)));
input_mus = (muspreconMatrix - min(muspreconMatrix(:))) / ...
            (max(muspreconMatrix(:)) - min(muspreconMatrix(:)));

target_mua = (muatargetMatrix - min(muatargetMatrix(:))) / ...
             (max(muatargetMatrix(:)) - min(muatargetMatrix(:)));
target_mus = (mustargetMatrix - min(mustargetMatrix(:))) / ...
             (max(mustargetMatrix(:)) - min(mustargetMatrix(:)));

% combining input and target into 2-channel format
inputData = cat(3, input_mua, input_mus);      % [res, res, 2, N]
targetData = cat(3, target_mua, target_mus);   % [res, res, 2, N]

% reshaping for datastore use
inputData = reshape(inputData, res, res, 2, []);
targetData = reshape(targetData, res, res, 2, []);

% number of images in the input data set
numImages = size(inputDataScaled, 3); 

inputDS = arrayDatastore(inputData, 'IterationDimension', 4);
targetDS = arrayDatastore(targetData, 'IterationDimension', 4);
dsTrain = combine(inputDS, targetDS);

%% building the network 

% built-in layers (https://se.mathworks.com/help/deeplearning/builtin-layers.html)
% layers are made to be similar to the U-net architecture in the U-net
% article 

% compare to this: https://github.com/prafful-kumar/Blurred-Image-Recognition/blob/main/Deblurring_U_NET(Keras).ipynb

% input layer
inputLayer = imageInputLayer([32 32 2], 'Name', 'input');

% encoder
enc1 = [
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'enc1_conv1')
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

% analyzeNetwork(lgraph) 

%% creating validation data

% splitting the input data for training and validation
numSamples = size(inputData, 4);
splitRatio = 0.8;
numTrain = round(splitRatio * numSamples);
idx = randperm(numSamples); 
trainIdx = idx(1:numTrain);
valIdx = idx(numTrain+1:end);

% training data
inputTrain = inputData(:,:,:,trainIdx);
targetTrain = targetData(:,:,:,trainIdx);
inputTrainDS = arrayDatastore(inputTrain, 'IterationDimension', 4);
targetTrainDS = arrayDatastore(targetTrain, 'IterationDimension', 4);
trainDS = combine(inputTrainDS, targetTrainDS);

% validation data
inputVal = inputData(:,:,:,valIdx);
targetVal = targetData(:,:,:,valIdx);
inputValDS = arrayDatastore(inputVal, 'IterationDimension', 4);
targetValDS = arrayDatastore(targetVal, 'IterationDimension', 4);
valDS = combine(inputValDS, targetValDS);

%% specifying the training options and training the net
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'InitialLearnRate', 1e-4, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'ValidationData', valDS, ...
    'ValidationFrequency', 250, ...
    'ExecutionEnvironment', 'gpu');


% training the net accoring to the options 
[diffNetRandInput, diffNetRandInputInfo] = trainNetwork(trainDS, lgraph, options);

% saving trained net 
save('DiffDeblurringUnet.mat', 'diffNetRandInput', 'lgraph');

%% test and display prediction for a random image from the set 
testIdx = randi([1 size(inputVal, 4)]);
testImg = inputVal(:, :, :, testIdx);
targetImg = targetVal(:, :, :, testIdx);

% making a prediction using the net 
predicted = predict(diffNetRandInput, testImg);

% displaying the predictions 
% figure;

% % mu_a (channel 1)
% subplot(2, 3, 1); imshow(testImg(:,:,1), []); title('Input \delta\mu_a');
% subplot(2, 3, 2); imshow(predicted(:,:,1), []); title('Predicted \delta\mu_a');
% subplot(2, 3, 3); imshow(targetImg(:,:,1), []); title('Ground Truth \delta\mu_a');
% 
% % mu_s (channel 2)
% subplot(2, 3, 4); imshow(testImg(:,:,2), []); title('Input \delta\mu_s');
% subplot(2, 3, 5); imshow(predicted(:,:,2), []); title('Predicted \delta\mu_s');
% subplot(2, 3, 6); imshow(targetImg(:,:,2), []); title('Ground Truth \delta\mu_s');
