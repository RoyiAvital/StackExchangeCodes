% StackExchange Signal Processing Q71822
% https://dsp.stackexchange.com/questions/71822
% Deconvolution of a 1D Time Domain Wave Signal Convolved with Series of Rect Signals
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     12/12/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Simulation Parameters

dataFile    = 'UserData.mat';
convShape   = CONVOLUTION_SHAPE_FULL;
noiseStd    = 0.00000003;

% TV
paramLambda     = 0.0025;
numIterations   = 350;
noiseStdFactor  = 100;


%% Generate Data

load(dataFile);

% Using Full Convolution: numElementsY = numElementsX + numElementsH - 1
numElementsX = size(vY, 1) - size(vH, 1) + 1;

mH = CreateConvMtx1D(vH, numElementsX, convShape);
mD = CreateConvMtx1D([1, -1], numElementsX, CONVOLUTION_SHAPE_VALID);

% Sanity Check
max(abs(mH * vX - vY)) %<! Should be very small


%% Naive Deconvolution

vXHat   = mH \ vY;
vXHatN  = mH \ (vY + (noiseStd * randn(size(vY, 1), 1)));


%% Display Naive Deconvolution

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100, 100, 760, 420]); %<! [x, y, width, height]
hAxes       = axes(); %<! [x, y, width, height]
hLineObj    = plot([vX, vXHat, vXHatN]);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(hLineObj(2), 'LineWidth', lineWidthThin, 'LineStyle', '--');
set(hLineObj(3), 'LineWidth', lineWidthThin, 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Naive Deconvolution'], ['Condition Number - ', num2str(condest(mH.' * mH))]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Smale Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
% set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
hLegend = ClickableLegend({['Ground Truth'], ['No Noise'], ['Noise STD - ', num2str(noiseStd)]});

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Total Variation Deconvolution

vXHat   = SolveLsTvAdmm([], full(mH), vY, full(mD), paramLambda, numIterations);
vXHatN  = SolveLsTvAdmm([], full(mH), (vY + (noiseStd * randn(size(vY, 1), 1))), full(mD), paramLambda, numIterations);
% vXHatNN = SolveLsTvAdmm([], full(mH), (vY + (noiseStdFactor * noiseStd * randn(size(vY, 1), 1))), full(mD), paramLambda, numIterations);


%% Display TV Deconvolution

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100, 100, 760, 420]); %<! [x, y, width, height]
hAxes       = axes(); %<! [x, y, width, height]
hLineObj    = plot([vX, vXHat, vXHatN]);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(hLineObj(2), 'LineWidth', lineWidthThin, 'LineStyle', '--');
set(hLineObj(3), 'LineWidth', lineWidthThin, 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Total Variation (TV) Deconvolution'], ['Condition Number - ', num2str(condest(mH.' * mH))]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Smale Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
% set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
hLegend = ClickableLegend({['Ground Truth'], ['No Noise'], ['Noise STD - ', num2str(noiseStd)]});

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

