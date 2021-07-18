% StackExchange Signal Processing Q51264
% https://dsp.stackexchange.com/questions/51264
% Detect Gradual Increase Before a Decrease in Noisy Data
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     04/04/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

dataFileName = 'Data.csv';

kernelRadius    = 5;
timeStd         = 1.5;
valueStd        = 2;


%% Generate Data

tData   = readtable(dataFileName);
vX      = tData{:, 2};

numSamples = size(vX, 1);


%% Apply Bilateral Filter

vY = BilateralFilter1D(vX, kernelRadius, timeStd, valueStd);


%% Display Results

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100, 100, 800, 600]);
hAxes       = axes(hFigure);
hLineObj    = plot(1:numSamples, [vX, vY]);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(hLineObj(2), 'LineWidth', lineWidthThin);
set(get(hAxes, 'Title'), 'String', {['Input Signal and Smoothed Signal']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Input Signal'], ['Smoothed (Bilateral Filter) Signal']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

