% StackExchange Signal Processing Q54730
% https://dsp.stackexchange.com/questions/54730
% Sequential Form of the Least Squares for Linear Least Squares Model
% References:
%   1.  A
% Remarks:
%   1.  Implementation of Vanilla Sequential Least Squares. In this
%       implementation no update for Covariance Matrix.
%   2.  The assumed model in Polynomial Model.
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     13/04/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

modelOrder      = 3; %<! Polynomial Order / Degree
numSamples      = 25;
numSamplesBatch = 5;
noiseStd        = 2;


%% Generate Data

mH      = repmat(linspace(0, 3, numSamples).', [1, modelOrder + 1]) .^ [0:modelOrder];
vTheta  = 3 * randn(modelOrder + 1, 1);
vN      = noiseStd * randn(numSamples, 1);
vX      = (mH * vTheta) + vN; %<! Measurements


%% Batch Least Squares Solution

vThetaEstLs = pinv(mH) * vX;


%% Sequential Least Squares

numIterations   = numSamples - numSamplesBatch;
vThetaInit      = pinv(mH(1:numSamplesBatch, :)) * vX(1:numSamplesBatch); %<! Initialization based on small batch

mThetaEstSequentialLs = zeros(modelOrder + 1, numIterations + 1);
mThetaEstSequentialLs(:, 1) = vThetaInit;

sampleIdx = numSamplesBatch;
for ii = 1:numIterations
    sampleIdx = sampleIdx + 1;
    mThetaEstSequentialLs(:, ii + 1) = UpdateSequentialLsModel(mH(1:sampleIdx, :), vX(sampleIdx), mThetaEstSequentialLs(:, ii));
end


%% Display Results

minVal = min(vX);
maxVal = max(vX);

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(mH(:, 2), [vX, mH * vThetaEstLs, mH * mThetaEstSequentialLs(:, 1)]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2:3), 'LineStyle', ':');
set(hAxes, 'YLim', [10 * floor(minVal / 10), 10 * ceil(maxVal / 10)]);
set(get(hAxes, 'Title'), 'String', {['Sequentail Least Squares Estimation vs. Batch Least Squares Estimation'], ['Sequential Least Squares - Estimation Based on Batch Mode of ', num2str(numSamplesBatch), ' First Samples and ', num2str(0) ' Sequential Samples']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Smaple Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Measurements'], ['Batch Least Squares Estimation'], ['Sequential Least Squares Estimation']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end

for ii = 1:numIterations
    figureIdx = figureIdx + 1;
    set(hLineSeries(3), 'YData', mH * mThetaEstSequentialLs(:, ii + 1));
    set(get(hAxes, 'Title'), 'String', {['Sequentail Least Squares Estimation vs. Batch Least Squares Estimation'], ['Sequential Least Squares - Estimation Based on Batch Mode of ', num2str(numSamplesBatch), ' First Samples and ', num2str(ii) ' Sequential Samples']}, ...
        'FontSize', fontSizeTitle);
    
    if(generateFigures == ON)
        saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    end
    
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

