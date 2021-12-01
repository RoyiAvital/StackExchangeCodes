% StackExchange Signal Processing Q79314
% https://dsp.stackexchange.com/questions/79314
% Image Segmentation Using Deep Learning
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     27/11/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

dataSetBaseUrl      = 'https://www.robots.ox.ac.uk/~vgg/data/pets/';
imageSetUrl         = 'data/images.tar.gz'; %<! https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
annotationSetUrl    = 'data/annotations.tar.gz'; %<! https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

imageSetFileName        = 'ImageSet.tar.gz';
annotationSetFileName   = 'annotationSet.tar.gz';

imageSetFolderName  = 'ImageSet';
maskSetFolderName   = 'MaskSet';

netCheckPointsFolderName = 'NetCheckPoints';

% Network parameters
numFiltersBase  = 16; %<! 16 In my Keras code
numBlocks       = 3;

% Input Data Parameters
% Data is with different sizes, resize all to 128x128
numRows     = 128; %<! 128 in my Keras code
numCols     = 128; %<! 128 in my Keras code
numChannels = 3;

classNames      = {'Pet', 'Background', 'Border'};
vPixelLabelId   = [1, 2, 3];

bacthSize = 96; %<! 64 in my Keras code


%% Generate Data

if(~exist(imageSetFileName, 'file'))
    websave(imageSetFileName, strcat(dataSetBaseUrl, imageSetUrl));
end
if(~exist(annotationSetFileName, 'file'))
websave(annotationSetFileName, strcat(dataSetBaseUrl, annotationSetUrl));
end

if(~exist(imageSetFolderName, 'dir'))
    untar(imageSetFileName);
    movefile('images', imageSetFolderName); %<! Fiels are in JPG format
end
if(~exist(maskSetFolderName, 'dir'))
    untar(annotationSetFileName);
    movefile(fullfile('annotations', 'trimaps'), maskSetFolderName); %<! Fiels are in JPG format
    rmdir('annotations', 's');
    sFiles = dir(strcat(maskSetFolderName, FILE_SEP, '.*.png'));

    for ii = 1:length(sFiles)
        delete(fullfile(sFiles(ii).folder, sFiles(ii).name));
    end

    sFiles = dir(strcat(maskSetFolderName, FILE_SEP, '*.mat'));
    for ii = 1:length(sFiles)
        delete(fullfile(sFiles(ii).folder, sFiles(ii).name));
    end
end

if(~exist(netCheckPointsFolderName, 'dir'))
    mkdir(netCheckPointsFolderName);
end


%% Build the Net

inputLayer      = imageInputLayer([numRows, numCols, numChannels], 'Name', 'InputLayer', 'Normalization', 'none');
dlnUnet         = BuildUNet(numBlocks, numFiltersBase);
softMaxLayer    = softmaxLayer('Name', 'SoftMaxLayer');
classLayer      = pixelClassificationLayer('Name', 'ClassificationLayer'); %<! Use weighing

dlnUnet = addLayers(dlnUnet, inputLayer);
dlnUnet = addLayers(dlnUnet, [softMaxLayer, classLayer]);
dlnUnet = connectLayers(dlnUnet, 'InputLayer', dlnUnet.Layers(1).Name);
dlnUnet = connectLayers(dlnUnet, dlnUnet.Layers(end - 3).Name, dlnUnet.Layers(end - 1).Name);

% Can't use as I couldn't find a way convert LayerGraph into DAGNetwork for
% prediction but training (assembleNetwork() doesn't work for non
% initizlied layers).
% save(strcat(netCheckPointsFolderName, FILE_SEP, 'net_checkpoint__0__', char(datetime('now', 'Format', 'yyyy_MM_dd_HH_mm_ss'))), 'dlnUnet');

% dlnUnet = unetLayers([numRows, numCols, numChannels], length(classNames));

figure();
plot(dlnUnet);
drawnow();


%% Data Set

imageDataStore  = imageDatastore(imageSetFolderName, 'FileExtensions', '.jpg');
imageDataStore.ReadFcn = @(x) ResizeImage(x, numRows, numCols);
maskDataStore   = pixelLabelDatastore(maskSetFolderName, classNames, vPixelLabelId, 'FileExtensions', '.png');
maskDataStore.ReadFcn = @(x) ResizeMask(x, numRows, numCols);

dsPets = combine(imageDataStore, maskDataStore);

% analyzeNetwork(dlnUnet.removeLayers('ClassificationLayer'), dlarray(rand([numRows, numCols, numChannels, 64]), 'SSCB'), 'TargetUsage', 'dlnetwork');
% drawnow();


%% Training

sTrainingOpt    = trainingOptions('adam', 'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', 'LearnRateDropPeriod', 2, 'LearnRateDropFactor', 0.96, ...
    'MaxEpochs', 15, 'MiniBatchSize', bacthSize, 'ExecutionEnvironment', 'gpu', ...
    'VerboseFrequency', 1, 'Plots', 'training-progress', 'CheckpointPath', netCheckPointsFolderName);
dlnUnet         = trainNetwork(dsPets, dlnUnet, sTrainingOpt);


%% Display Results

% mI = imageDataStore.readimage(1);
% mC = maskDataStore.readimage(1);
% mCI = zeros(numRows, numCols);
% 
% for ii = 1:length(classNames)
%     mCI(mC == classNames{ii}) = vPixelLabelId(ii);
% end
% 
% sNetFiles = dir(strcat(netCheckPointsFolderName, FILE_SEP, '*.mat'));
% for ii = 1:length(sNetFiles)
%     sNetPred    = load(fullfile(netCheckPointsFolderName, sNetFiles(ii).name));
%     netPred     = sNetPred.net;
%     mCPred      = semanticseg(mI, netPred);
%     figure();
%     image(mCPred);
% end

% figureIdx = figureIdx + 1;
% 
% hFigure = DisplayComparisonSummary(numIterations, mObjFunValMse, mSolMse, cLegendString, figPosLarge, lineWidthNormal, fontSizeTitle, fontSizeAxis);
% 
% if(generateFigures == ON)
%     % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
%     print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
% end
% 
% 
% figureIdx = figureIdx + 1;
% 
% hFigure = figure('Position', figPosLarge);
% hAxes   = axes(hFigure);
% hLineObj = plot(1:numSamples, [vW, vY, vX]);
% set(hLineObj(1), 'LineWidth', lineWidthNormal);
% set(hLineObj(2), 'LineStyle', 'none', 'Marker', '*');
% % set(hLineObj(3), 'LineWidth', lineWidthThin, 'LineStyle', ':');
% set(hLineObj(3), 'LineStyle', 'none', 'Marker', 'x');
% set(get(hAxes, 'Title'), 'String', {['Signals']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
%     'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['Ground Truth'], ['Input Noisy Samples'], ['TV Estimation']});
% 
% if(generateFigures == ON)
%     % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
%     print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
% end

function [ mO ] = ResizeImage( fileName, numRows, numCols )

mO = im2single(imresize(imread(fileName), [numRows, numCols]));

% Some images are 1 channel
if(size(mO, 3) == 1)
    mO = repmat(mO, [1, 1, 3]);
end


end


function [ mO ] = ResizeMask( fileName, numRows, numCols )

mO = imresize(imread(fileName), [numRows, numCols]);
if(size(mO, 3) ~= 1)
    mO = mO(:, :, 1);
end


end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

