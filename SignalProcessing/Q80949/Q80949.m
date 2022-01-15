% StackExchange Signal Processing Q80949
% https://dsp.stackexchange.com/questions/80949
% Image Clustering Using Linear Discriminant Analysis (LDA) Compared to t-SNE / UMAP
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     10/01/2022
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

dataSetBaseUrl      = 'http://yann.lecun.com/exdb/mnist/';
imageSetUrl         = 'train-images-idx3-ubyte.gz'; %<! http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
annotationSetUrl    = 'train-labels-idx1-ubyte.gz'; %<! http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

imageSetFileName        = 'ImageSet';
annotationSetFileName   = 'AnnotationSet';

fileExt = '.gz';

% Parsing Parameters
trimImg     = 1;
sclaeImg    = 1;

% t-SNE Parameters
numFeatures = 2;
numPcaComp = 50;


%% Generate / Load Data

if(~exist(strcat(imageSetFileName, fileExt), 'file'))
    websave(strcat(imageSetFileName, fileExt), strcat(dataSetBaseUrl, imageSetUrl));
end
if(~exist(strcat(annotationSetFileName, fileExt), 'file'))
    websave(strcat(annotationSetFileName, fileExt), strcat(dataSetBaseUrl, annotationSetUrl));
end

if(~isfile(imageSetFileName))
    gunzip(strcat(imageSetFileName, fileExt));
end
if(~isfile(annotationSetFileName))
    gunzip(strcat(annotationSetFileName, fileExt));
end

[tImg, vLabel] = ParseMnistData(imageSetFileName, annotationSetFileName, trimImg, sclaeImg);

numRows = size(tImg, 1);
numCols = size(tImg, 2);
numImg = size(tImg, 3);


%% Analysis by Linear Discriminant Analysis

mX = reshape(tImg, numRows * numCols, numImg).';

mW = LinearDiscriminantAnalysis(mX, vLabel);
mF = mX * mW(:, 1:numFeatures);

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
hSctrGrp = gscatter(mF(:, 1), mF(:, 2), vLabel);
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


%% Analysis by t-SNE

% Low dimensionality features by t-SNE
mF = tsne(reshape(tImg, numRows * numRows, numImg).', 'Algorithm', 'barneshut', 'NumDimensions', 2, 'NumPCAComponents', 50);

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
hSctrGrp = gscatter(mF(:, 1), mF(:, 2), vLabel);
set(get(hAxes, 'Title'), 'String', {['Clustering by 2 Features of t-SNE']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['t-SNE Feature #1']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['t-SNE Feature #2']}, ...
    'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['Ground Truth'], ['Input Noisy Samples'], ['TV Estimation']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

