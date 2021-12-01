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

% Network parameters
numFiltersBase  = 4; %<! 16 In my Keras code
numBlocks       = 3;

% Input Data Parameters
% Data is with different sizes, resize all to 128x128
numRows     = 64; %<! 128 in my Keras code
numCols     = 64; %<! 128 in my Keras code
numChannels = 3;

classNames      = {'Pet', 'Background', 'Border'};
vPixelLabelId   = [1, 2, 3];

bacthSize = 16; %<! 64 in my Keras code


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
    rmdir('annotations\', 's');
    sFiles = dir(strcat(maskSetFolderName, '/.*.png'));

    for ii = 1:length(sFiles)
        delete(fullfile(sFiles(ii).folder, sFiles(ii).name));
    end

    sFiles = dir(strcat(maskSetFolderName, '/*.mat'));
    for ii = 1:length(sFiles)
        delete(fullfile(sFiles(ii).folder, sFiles(ii).name));
    end
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


dlnUnet = unetLayers([numRows, numCols, numChannels], length(classNames));

figure();
plot(dlnUnet);
drawnow();


%% Data Set

imageDataStore  = imageDatastore(imageSetFolderName, 'FileExtensions', '.jpg');
maskDataStore   = pixelLabelDatastore(maskSetFolderName, classNames, vPixelLabelId, 'FileExtensions', '.png');
maskDataStore.ReadFcn = @(x) imresize(imread(x), [numRows, numCols]);

dsPets = combine(imageDataStore.transform(@(x) imresize(im2single(x), [numRows, numCols])), maskDataStore);

% analyzeNetwork(dlnUnet.removeLayers('ClassificationLayer'), dlarray(rand([numRows, numCols, numChannels, 64]), 'SSCB'), 'TargetUsage', 'dlnetwork');
% drawnow();


%% Training

sTrainingOpt    = trainingOptions('adam', 'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 20, 'MiniBatchSize', bacthSize, 'ExecutionEnvironment', 'cpu', ...
    'VerboseFrequency', 1, 'Plots', 'training-progress');
dlnUnet         = trainNetwork(dsPets, dlnUnet, sTrainingOpt);


%% Display Results

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


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

