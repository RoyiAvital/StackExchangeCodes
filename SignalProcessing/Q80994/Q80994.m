% StackExchange Signal Processing Q80994
% https://dsp.stackexchange.com/questions/80994
% Building a Pipeline for Image Classification / Clustering Tasks with Features Extractor (Example on MNIST Data)
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     29/12/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

% Run 'PreProcess.m' before this file
dataSetFileName = 'MNISTDataSet.mat';

% Local Binary Pattern (LBP) Features Extractor Parameters
numNeighbors    = 8; %<! Default is 8
vCellSize       = [7, 7]; %<! Images are 28 x 28 -> 4 x 4 cell.

% LDA Dimensionality Reduction Parameters
numLdaFeatures = 10;


%% Generate / Load Data

% Loading data into: tTrainSetImg, vTrainSetLabel, tTestSetImg, vTestSetLabel
load(dataSetFileName);

[numRows, numCols, numTrainImg] = size(tTrainSetImg);
numTestImg                      = size(tTestSetImg, 3);


%% Feature Extraction

numFeaturesPerImg = ((numNeighbors * (numNeighbors - 1)) + 3) * (floor(numRows / vCellSize(1)) * floor(numCols / vCellSize(2)));

mF = zeros(numTrainImg, numFeaturesPerImg);

for ii = 1:numTrainImg
    mF(ii, :) = extractLBPFeatures(tTrainSetImg(:, :, ii), 'NumNeighbors', numNeighbors, 'CellSize', vCellSize);
    disp(['Processed Image #', num2str(ii, '%04d'), ' Out of ', num2str(numTrainImg, '%04d'), ' Images.'])
end


%% Dimensionality Reduction by LDA (Supervised Linear Dimensionality Reduction)

mW = LinearDiscriminantAnalysis(mF, vTrainSetLabel);
mW = mW(:, 1:numLdaFeatures);

mF = mF * mW;

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
hSctrGrp = gscatter(mF(:, 1), mF(:, 2), vTrainSetLabel);
set(get(hAxes, 'Title'), 'String', {['Clustering by 2 Features of Linear Discriminant Analysis (LDA)']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['LDA Feature #1']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['LDA Feature #2']}, ...
    'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['Ground Truth'], ['Input Noisy Samples'], ['TV Estimation']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Classification by SVM with Gaussian Kernel

hSvmTemplate = templateSVM('KernelFunction', 'gaussian', 'PolynomialOrder', [], ...
    'KernelScale', 3.2, 'BoxConstraint', 1, 'Standardize', true);

classificationSVM = fitcecoc(mF, vTrainSetLabel, 'Learners', hSvmTemplate, ...
    'Coding', 'onevsone', 'ClassNames', unique(vTrainSetLabel));


%% Classification of the Test Set

mF = zeros(numTestImg, numFeaturesPerImg);

for ii = 1:numTestImg
    mF(ii, :) = extractLBPFeatures(tTestSetImg(:, :, ii), 'NumNeighbors', numNeighbors, 'CellSize', vCellSize);
    disp(['Processed Image #', num2str(ii, '%04d'), ' Out of ', num2str(numTestImg, '%04d'), ' Images.'])
end

mF = mF * mW; %<! Dimensionality Recution by the learned transofrmation
vPredLabel = predict(classificationSVM, mF);

mC = confusionmat(vTestSetLabel, vPredLabel, 'Order', unique(vTrainSetLabel));


figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
hConfusionMtx = confusionchart(mC, 'Title', {['MNIST Data Set Classification - Confusion Matrix'], ['Classifcation Success Rate: ', num2str(100 * mean(vTestSetLabel == vPredLabel)), '%']});
% set(get(hAxes, 'Title'), 'String', {['MNIST Data Set Classification - Confusion Matrix']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hAxes, 'XLabel'), 'String', {['t-SNE Feature #1']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hAxes, 'YLabel'), 'String', {['t-SNE Feature #2']}, ...
%     'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['Ground Truth'], ['Input Noisy Samples'], ['TV Estimation']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

disp(['Classifcation Success Rate: ', num2str(100 * mean(vTestSetLabel == vPredLabel)), '%']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

