% Convolution Gradient Analysis
% References:
%   1.  Deconvolve of a 1D Signal with Known Kernel (Square Wave) - https://dsp.stackexchange.com/questions/51460.
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     24/08/2018
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;
DIFF_MODE_COMPLEX   = 4;


%% Simulation Parameters

numSamples      = 1000;
kernelRadius    = 4;

difMode = DIFF_MODE_COMPLEX;
epsVal  = 1e-9;


%% Generate Data

kernelLength = (2 * kernelRadius) + 1;

vX = randn(numSamples, 1);
vH = randn(kernelLength, 1);

vY = conv2(vX, vH, 'valid');

hObjFun = @(vX) 0.5 * sum( (conv2(vX, vH, 'valid') - vY) .^ 2 );


%% Analysis - Numeric Gradient vs. Analytic Gradient

vG      = conv2((conv2(vX, vH, 'valid') - vY), vH(end:-1:1), 'full');
vGRef   = CalcFunGrad(vX, hObjFun, difMode, epsVal);


%% Analysis

maxDev = max(abs(vG - vGRef));
disp([' ']);
disp(['Maximum Absulote Deviation (MAD) - ', num2str(maxDev)]);
disp([' ']);


%% Display Results

figureIdx = figureIdx + 1;

% hFigure = figure('Position', figPosLarge);
% hAxes   = axes();
% set(hAxes, 'NextPlot', 'add');
% hLineSeries = plot(mX(1, :), mX(3, :));
% set(hLineSeries, 'LineWidth', lineWidthThin, 'Marker', '*', 'Color', mColorOrder(1, :));
% hLineSeries = plot(mY(1, :), mY(2, :));
% set(hLineSeries, 'LineStyle', 'none', 'Marker', '*', 'Color', mColorOrder(2, :));
% hLineSeries = plot(mXEst(1, :), mXEst(3, :));
% set(hLineSeries, 'LineStyle', 'none', 'Marker', '*', 'Color', mColorOrder(3, :));
% set(hAxes, 'DataAspectRatio', [1, 1, 1]);
% set(hAxes, 'Xlim', [0, 2000], 'YLim', [0, 2000]);
% set(get(hAxes, 'Title'), 'String', {['Extended Kalman Estimation - Cartesian Coordinate Model and Polar Coordinate Measurement']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hAxes, 'XLabel'), 'String', {['x [Meters]']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hAxes, 'YLabel'), 'String', {['y [Meters]']}, ...
%     'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['Ground Truth'], ['Measurement'], ['Estimation']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

