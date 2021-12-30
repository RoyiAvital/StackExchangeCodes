% StackExchange Signal Processing Q80767
% https://dsp.stackexchange.com/questions/80767
% Unsupervised Clustering of Images - How?
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


%% Analysis by Mean and Variance

mF = zeros(numImg, 2);

for ii = 1:numImg
    mF(ii, 1) = mean(tImg(:, :, ii), 'all');
    mF(ii, 2) = var(tImg(:, :, ii), 0, 'all');
end

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
hSctrGrp = gscatter(mF(:, 1), mF(:, 2), vLabel);
set(get(hAxes, 'Title'), 'String', {['Clustering by Mean and Variance']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Mean']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Variance']}, ...
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

