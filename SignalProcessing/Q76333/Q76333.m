% StackExchange Signal Processing Q76333
% https://dsp.stackexchange.com/questions/76333
% Deblurring 1D data using Direct Inverse Filtering
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     18/07/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Simulation Parameters

dataFileName    = 'Data.csv';
vH              = [1; 4; 6; 4; 1] / 16; %<! Regularization
noiseStd        = 2;
paramLambda     = 0.25;


%% Generate Data

tData       = readtable(dataFileName);
vX          = tData{:, 1}; %<! Ground Truth
numSamples  = size(vX, 1);
vY          = tData{:, 2}; %<! Seem not to be a filtered version of vX
% vY          = conv(vX, vH, 'same') + (noiseStd * randn(numSamples, 1)); %<! Synthetic
mH          = CreateConvMtx1D(vH, size(vY, 1), CONVOLUTION_SHAPE_SAME);
% vB          = firls(50, [0, 0.03, 0.05, 1],[1, 1, 0, 0], [1, 1]);
% fvtool(vB, 1, 'OverlayedAnalysis', 'phase')
% vYY = conv(vY, vB, 'same');
% figure(); plot([vX, vY, vYY]);


%% Deconvolution (Without Regularization)

vXEstWoReg = mH \ vY;


%% Deconvolution (With Regularization)

% This is not an optimized way to solve this. Probably a better way to do
% so is use LSMB (See https://math.berkeley.edu/~ehallman/lsmb/).
vXEstWReg = (mH.' * mH + (paramLambda * paramLambda * speye(numSamples))) \ (mH.' * vY);


%% Display Results

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100, 100, 800, 600]);
hAxes       = axes(hFigure);
hLineObj    = plot(1:numSamples, [vX, vY]);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Data']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth Signal'], ['Measured Signal']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

figureIdx = figureIdx + 1;

vH(numSamples) = 0;

hFigure     = figure('Position', [100, 100, 800, 600]);
hAxes       = axes(hFigure);
PlotDft([vX, vY, vH], 1, 'normalizDataFlag', 1, 'removeDc', 1, 'plotTitle', 'The DFT of the Data and Filter', 'plotLegendFlag', 1,'plotLegend', {['GT Signal'], ['Measured Signal'], ['Filter']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100, 100, 800, 600]);
hAxes       = axes(hFigure);
hLineObj    = plot(1:numSamples, [vX, vY, vXEstWoReg, vXEstWReg]);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Comparison of Deconvolution Methods']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['Input Signal'], ['Estimated without Regularization'], ['Estimated with Regularization']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

