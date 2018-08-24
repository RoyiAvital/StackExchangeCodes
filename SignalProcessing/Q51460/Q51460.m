% Mathematics Q51460
% https://dsp.stackexchange.com/questions/51460
% Deconvolution of a 1D Signal with Known Kernel (Square Wave)
% References:
%   1.  A
% Remarks:
%   1.  CSV Data is given at https://dsp.stackexchange.com/questions/51460/#comment102211_51460.
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     24/08/2018
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

kernelRadius    = 2;
kernelValue     = 1;

paramLambda = 0.0;

numIterations   = 500000;
stepSize        = 0.001;


%% Generate Data

% Reading CSV Data
% First 2 rows should be ignored.
% Columns:
%   1. X Grid Data.
%   2. Y Value Data.
%   3. NA.
%   4. Kernel X Grid Data.
%   5. Kernel Y Value Data.
% In columns (4, 5) one should ignore sample 5 and onward (Including 5).
mData = csvread('SignalsData.csv', 2);

kernelLength = (2 * kernelRadius) + 1;

vY = mData(:, 2);
vH = kernelValue * ones(kernelLength, 1);

vObjVal = zeros(numIterations + 1, 1);
hObjFun = @(vX) 0.5 * sum((conv(vX, vH, 'valid') - vY) .^ 2);


%% Deconvolution Process

vX = zeros(size(vY, 1) + (2 * kernelRadius), 1);
% vX = [zeros(kernelRadius, 1); vY; zeros(kernelRadius, 1)];
vObjVal(1) = hObjFun(vX);

for ii = 1:numIterations
    vG = conv2((conv2(vX, vH, 'valid') - vY), vH(end:-1:1), 'full') + (paramLambda * vX);
    vX = vX - (stepSize * vG);
    
    vObjVal(ii + 1) = hObjFun(vX);
end


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes();
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot([vY, conv(vX, vH, 'valid')]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Deconvolution with Box Blur']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Smaple Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['$ h \ast x $'], ['Input Signal']}, 'Interpreter', 'latex');

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end

vY = [zeros(kernelRadius, 1); vY; zeros(kernelRadius, 1)];

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes();
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot([vX, vY]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Deconvolution with Box Blur']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Smaple Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Deconvolution Signal'], ['Input Signal']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes();
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(0:numIterations, vObjVal);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Deconvolution Objective Value']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Interation Number']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Function Value']}, ...
    'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['Deconvolution Signal'], ['Input Signal']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes();
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(0:numIterations, 10 * log10(vObjVal));
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Deconvolution Objective Value']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Interation Number']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Function Value [dB]']}, ...
    'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['Deconvolution Signal'], ['Input Signal']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

