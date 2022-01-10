% StackExchange Signal Processing Q80949
% https://dsp.stackexchange.com/questions/80949
% Image Clustering Using Linear Discriminant Analysis (LDA) Compared to t-SNE / UMAP
% Testing the implementation of LDA
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

dataSetUrl      = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'; %<! Basicaly a CSV file
dataSetFileName = 'wine.data';

% LDA Parameters
numFeatures = 2;


%% Generate / Load Data

if(~exist(dataSetFileName, 'file'))
    websave(dataSetFileName, dataSetUrl);
end

mWineData = readmatrix(dataSetFileName, 'FileType', 'text');
mX = mWineData(:, 2:end);
vY = mWineData(:, 1);


%% Analysis by LDA

mW = LinearDiscriminantAnalysis(mX, vY);
mF = mX * mW(:, 1:numFeatures);

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
hSctrGrp = gscatter(mF(:, 1), mF(:, 2), vY);
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


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

