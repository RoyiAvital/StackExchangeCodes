% StackExchange Signal Processing Q97923
% https://dsp.stackexchange.com/questions/97923
% Robust Extraction of Local Peaks in Noisy Signal with a Trend
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     21/06/2025
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants

MODEL_SSA = 1; %<! Singular Spectrum Analysis for Time Series
MODEL_STL = 2; %<! STL: A Seasonal Trend Decomposition Procedure Based on Loess

%% Parameters

% Data
csvFileName = 'Signal.csv';

numSeasComp = 2; %<! For SSA Decomposition
vP          = 10:20; %<! For STL Periods

% Model
decModel = MODEL_STL; %<! Decomposition Mode


%% Generate / Load Data

tA = readtable(csvFileName);
vX = tA.Time;
vY = tA.Signal;


%% Analysis

switch(decModel)
    case(MODEL_SSA)
        [vT, mS, vR]  = trenddecomp(vY, 'NumSeasonal', numSeasComp);
        numSeasSignal = numSeasComp;
        decModelStr   = 'SSA';
    case(MODEL_STL)
        [vT, mS, vR]  = trenddecomp(vY, 'stl', vP);
        numSeasSignal = length(vP);
        decModelStr   = 'STL';
end

numSignals = numSeasSignal + 3;


%% Display Results

cL = cell(1, 3 + numSeasSignal);
cL{1} = 'Data';
cL{2} = 'Trend';
jj = 1;
for ii = 3:(3 + numSeasSignal - 1)
    cL{ii} = ['Seasonality ', num2str(jj, '%02d')];
    jj = jj + 1;
end
cL{end} = 'Remainder';

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(vX, [vY, vT, mS, vR]);
for ii = 1:numSignals
    set(hLineObj(ii), 'DisplayName', cL{ii});
end
set(hLineObj, 'LineWidth', 1.5);

set(get(hAxes, 'Title'), 'String', {['Data and Decomposition (', decModelStr, ')']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);

hLegend = ClickableLegend('Location', 'SouthWest');

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(vX, [vY, vT, vY - vT]);
set(hLineObj, 'LineWidth', 1.5);
for ii = 1:2
    vColor = get(hLineObj(ii), 'Color');
    vColor(end + 1) = 0.35;
    set(hLineObj(ii), 'Color', vColor);
end

set(get(hAxes, 'Title'), 'String', {['Data, Trend and Detrend (', decModelStr, ')']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);

hLegend = ClickableLegend({'Signal', 'Trend', 'Detrend'}, 'Location', 'SouthWest');

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Auxiliary Functions



%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

