% Final Project
close all;
clear;
clc;

%__________XCEPTION________

% ___WORKING WITH DATA_____

% Loading data

addpath("C:\Users\sasha\OneDrive\Desktop\BrainTumor_Classification\med_data");

train_data = "C:\Users\sasha\OneDrive\Desktop\BrainTumor_Classification\med_data\Training"

test_data = "C:\Users\sasha\OneDrive\Desktop\BrainTumor_Classification\med_data\Testing"


img_data_train_all = imageDatastore(train_data, ...
    "IncludeSubfolders", true, ...
    "LabelSource","foldernames");

img_data_test = imageDatastore(test_data, ...
    "IncludeSubfolders", true, ...
    "LabelSource","foldernames");

% splitting given training dataset into 70% training and 30% validation
[img_data_train, img_data_val] = splitEachLabel(img_data_train_all, 0.7, "randomized");
% the input size changes for Xception
inp_size = [299 299 3];

% Load images from the augmented image datastores
[train_im, train_lbl] = datastoreTo4D(img_data_train, inp_size);
[test_im, test_lbl] = datastoreTo4D(img_data_test, inp_size);
[val_im, val_lbl] = datastoreTo4D(img_data_val, inp_size);

% ___LABELING DATA_____

% Convert labels to categorical for easier handling
train_lbl = categorical(train_lbl);
test_lbl = categorical(test_lbl);
val_lbl = categorical(val_lbl);


train_im = uint8(train_im);
test_im = uint8(test_im);
val_im = uint8(val_im);

% 10%

% Number of samples for 10%
numTrainSamples = floor(0.1 * size(train_im, 4));

% Randomly sample indices
rng('default'); % for reproducibility
idx = randperm(size(train_im, 4), numTrainSamples);

% Subsample train images and labels
train_im = train_im(:, :, :, idx);
train_lbl = train_lbl(idx);


% augmentedImageDatastores

in_size = [size(train_im,1), size(train_im, 2), size(train_im, 3)];

augm_train = augmentedImageDatastore(in_size, train_im, train_lbl);
augm_test = augmentedImageDatastore(in_size, test_im, test_lbl);
augm_val = augmentedImageDatastore(in_size, val_im, val_lbl);


% ___SETTING XCEPTION___

net = xception;

lgraph = layerGraph(net);

classes_num = numel(categories(train_lbl));

newFCLayer = fullyConnectedLayer(classes_num, ...
    "Name", "new_fc", ...
    "WeightLearnRateFactor", 10, ...
    "BiasLearnRateFactor", 10);

newClassLayer = classificationLayer("Name", "new_class_out");


lgraph = replaceLayer(lgraph, 'predictions', newFCLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);



inp_size = net.Layers(1).InputSize;

augm_train = augmentedImageDatastore(inp_size, train_im, train_lbl);
augm_test = augmentedImageDatastore(inp_size, test_im, test_lbl);
augm_val = augmentedImageDatastore(inp_size, val_im, val_lbl);

% ___TRAINING___

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augm_val, ...
    "ValidationFrequency", 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network using the augmented image datastore
trainedNet = trainNetwork(augm_train, lgraph, options);


% ___EVALUATION___

[YPred, scores] = classify(trainedNet, augm_test);

classes = categories(test_lbl);
num_class = numel(classes);

AUCx = zeros(num_class,1);

figure; hold on;

for k = 1:num_class
    posClass = classes{k};
    [X, Y, ~, AUC] = perfcurve(test_lbl, scores(:,k), posClass);
    AUCx(k) = AUC;

    plot(X,Y, 'DisplayName', posClass);
end

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves Xception');
legend('show');
hold off;

% ___HELPER FUNCTION FOR DATA TRANSFORMATION____

function [X, Y] = datastoreTo4D(ds, inp_size)
    % Converts an ImageDatastore into a 4D uint8 array X and label vector Y.

    % get the amount of files
    img_num = numel(ds.Files);

    % Initialize 4D array
    X = zeros([inp_size img_num], 'uint8');
    Y = ds.Labels;

    for i = 1:img_num
        I = readimage(ds, i);

        % Convert color model
        if size(I,3) == 1
            I = cat(3, I, I, I);
        end

        % change to correct size
        I = imresize(I, inp_size(1:2));

        % convert to uint8 type
        I = im2uint8(I);

        % add to the 4D array
        X(:,:,:,i) = I;
    end
end


% ___DISPLAYING RESULTS___

% Display AUC table
AUC_Table = table(classes, AUCx, 'VariableNames', {'Class','AUC'});
disp(AUC_Table);
