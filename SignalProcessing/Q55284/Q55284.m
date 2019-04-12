% StackExchange Signal Processing Q55284
% https://dsp.stackexchange.com/questions/55284
% Deconvolution of Synthetic 1D Signals - How To?
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     07/02/2019
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

numIterations   = 100000;
stepSize        = 5e-2;

kernelStd               = 2.45;
kernelToRadiusFactor    = 7;
numSamplesSignal        = 2.5e3;


%% Generate Data

% Generating Convolution Kernel
kernelRadius = ceil(kernelStd * kernelToRadiusFactor);
vX = [-kernelRadius:kernelRadius];
vX = vX(:);
vH = GenerateGaussianCurveMembershipFunction(vX, 0, kernelStd); % + GenerateGaussianCurveMembershipFunction(vX, 2.5, 0.45) + GenerateGaussianCurveMembershipFunction(vX, 3.7, 0.25);
vH = vH / sum(vH);

% Generating Signal - Rectangle Train
vX = linspace(0, 30, numSamplesSignal);
vX = vX(:);
vS = zeros(numSamplesSignal, 1);

vS(mod(round(vX), 2) == 1) = 1;


%% Convolution & Deconvolution

% The convolution result
vC = conv2(vS, vH, 'same');

mH = CreateConvMtx1D(vH, numSamplesSignal, CONVOLUTION_SHAPE_SAME);
vSR = pinv(full(mH)) * vC; %<! The Deconvolution Result



%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = subplot(4, 1, 1);
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vX, vS);
set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Input Signal']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['DFT'], ['FFT']});

hAxes   = subplot(4, 1, 2);
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot([-kernelRadius:kernelRadius], vH);
set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Convolution Kernel']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['DFT'], ['FFT']});

hAxes   = subplot(4, 1, 3);
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vX, vC);
set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Convolution Output Signal']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['DFT'], ['FFT']});

hAxes   = subplot(4, 1, 4);
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vX, [vS, vSR]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2), 'LineStyle', ':');
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Input Signal vs. Reconstructed Signal (Deconvolution)']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['DFT'], ['FFT']});

if(generateFigures == ON)
    saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

