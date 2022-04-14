% StackExchange Signal Processing Q59325
% https://dsp.stackexchange.com/questions/59325
% Learning the Coefficients of Auto Regressive Model Using Least Mean
% Squares Filter for Signal Prediction
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     14/04/2022
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Constants


%% Simulation Parameters

% Signal Generation
numSamples  = 2000;
numSignals = 30;

% Estimaiton
arModelOrder = 101;
numSamplesEst = round(0.75 * numSamples);


%% Generate / Load Data

vA = rand(numSignals, 1) + 0.35;
vF = 5 * rand(numSignals, 1) + 1;
vP = pi * rand(numSignals, 1);

vT = linspace(0, 5, numSamples).';
vX = zeros(numSamples, 1);

for ii = 1:length(vA)
    vX(:) = vX(:) + vA(ii) * sin(2 * pi  * vF(ii) * vT + vP(ii));
end

vW = zeros(arModelOrder, 1); %<! LMS initialization


%% LMS

vD = vX((arModelOrder + 1):(arModelOrder + 1 + numSamplesEst));
vY = vX(1:(numSamplesEst + 1));

vWW = LmsFilter(vW, vY, vD, arModelOrder, numSamplesEst, 5e-4, OFF);

vYY = filter(1, [1; vWW], vX); %<! Prediction by the AR Model as estimated by the LMS


%% Analysis

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(vT(1:numSamplesEst), vX(1:numSamplesEst));
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = plot(vT((numSamplesEst + 1):end), vX((numSamplesEst + 1):end));
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Linear Combination of ', num2str(numSignals), ' Sine Signals']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Time Index [Sec]']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Input Signal: Samples for LMS'], ['Input Signal: Samples to Predict']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(vT, vX);
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = plot(vT, vYY);
set(hLineObj, 'LineWidth', lineWidthNormal, 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['LMS Filter Prediction'], ['The RMSE of the Estimation: ', num2str(sqrt(mean((vYY - vX) .^ 2)))]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Time Index [Sec]']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Input Signal'], ['Predicted Signal (Model Order: ', num2str(arModelOrder), ')']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

