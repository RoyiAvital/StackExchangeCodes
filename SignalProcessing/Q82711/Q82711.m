% StackExchange Signal Processing Q82711
% https://dsp.stackexchange.com/questions/82711
% A Machine Learning Algorithm to Replace Matched Filter?
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     26/04/2022
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

% Data
dataFile = 'Data.mat';

% Net
bacthSize = 2048;
netCheckPointsFolderName = 'NetCheckPoints';


%% Generate / Load Data

load(dataFile);

numFeatures     = size(mData, 1);
numSamplesSig   = size(vSignal, 1);
numClasses      = length(unique(vLabels(:)));


%% Build the Net

dlNet = [ ...
    sequenceInputLayer(numFeatures)
    convolution1dLayer(numSamplesSig, 20, Padding = "causal")
    reluLayer
    layerNormalizationLayer
    convolution1dLayer(ceil(numSamplesSig / 2), 40, Padding = "causal")
    reluLayer
    layerNormalizationLayer
    convolution1dLayer(5, 40, Padding = "causal")
    reluLayer
    layerNormalizationLayer
    globalAveragePooling1dLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

dlNet = [ ...
    sequenceInputLayer(numFeatures)
    convolution1dLayer(numSamplesSig, 20, Padding = "causal")
    reluLayer
    convolution1dLayer(ceil(numSamplesSig / 2), 40, Padding = "causal")
    reluLayer
    convolution1dLayer(5, 40, Padding = "causal")
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

dlNet = layerGraph(dlNet);

figure();
plot(dlNet);
drawnow();


%% Data Set

xDataStore = arrayDatastore(mData, 'IterationDimension', 2);
yDataStore = arrayDatastore(categorical(vLabels));

dsSignal = combine(xDataStore, yDataStore);


%% Training

sTrainingOpt    = trainingOptions('adam', 'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', 'LearnRateDropPeriod', 2, 'LearnRateDropFactor', 0.96, ...
    'MaxEpochs', 15, 'MiniBatchSize', bacthSize, 'ExecutionEnvironment', 'gpu', ...
    'VerboseFrequency', 1, 'Plots', 'training-progress', 'CheckpointPath', netCheckPointsFolderName);
dlNet           = trainNetwork(dsSignal, dlNet, sTrainingOpt);


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


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

