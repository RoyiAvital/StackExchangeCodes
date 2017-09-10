% Cross Validated Q122153
% https://stats.stackexchange.com/questions/122153
% Least Squares Estimation of Poisson Parameter
% References:
%   1.  Poisson Distribution Wikipedia - https://en.wikipedia.org/wiki/Poisson_distribution.
%   2.  Poisson Distribution (MATLAB)- https://www.mathworks.com/help/stats/poisson-distribution.html.
%   3.  Poisson Random Numbers (MATLAB)- https://www.mathworks.com/help/stats/poissrnd.html.
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     10/09/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

paramLambda = 0.75;
numSamples  = 1000;


%% Algorithm Parameters

gridStartVal    = 0.01;
gridEndVal      = 2;
numGridSamples  = 2000;


%% Generate Data

vX              = randi([1, 10], [numSamples, 1]); %<! Known
vParamLambda    = paramLambda * vX;
vDataSamples    = poissrnd(vParamLambda, [numSamples, 1]);

vLambdaGrid     = linspace(gridStartVal, gridEndVal, numGridSamples);


%% Maximum Likelihood Estimator of Lambda - Brute Force

vLogLikelihood  = zeros([numGridSamples, 1]);

for ii = 1:numGridSamples
    currLambda = vLambdaGrid(ii);
    vLogLikelihood(ii) = log(currLambda) * sum(vDataSamples) + sum(log(vX)) - currLambda * sum(vX) - sum(log(factorial(vDataSamples)));
end


%% Maximum Likelihood Estimator of Lambda - Closed Form

paramLambdaMle = sum(vDataSamples) / sum(vX);


%% Least Squares Estimator of Lambda - Brute Force

vLsLikelihood  = zeros([numGridSamples, 1]);

for ii = 1:numGridSamples
    currLambda          = vLambdaGrid(ii);
    vLsLikelihood(ii)   = sum( (vDataSamples - (currLambda * vX)) .^ 2 );
end


%% Least Squares Estimator of Lambda - Closed Form

paramLambdaLs = sum(vDataSamples .* vX) / sum(vX .^ 2);


%% Analysis

hFigure     = figure('Position', figPosLarge);
hAxes       = subplot(2, 1, 1);
set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vLambdaGrid, vLogLikelihood);
set(hLineSeries, 'LineWidth', lineWidthNormal);
hLineSeries = plot([paramLambda, paramLambda], get(hAxes, 'YLim'));
set(hLineSeries, 'LineStyle', ':');
hLineSeries = plot([paramLambdaMle, paramLambdaMle], get(hAxes, 'YLim'));
set(hLineSeries, 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Log Likelihood Objective Function Value vs. \lambda Hypothesis']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', '\lambda', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Log Likelihood Objective Function'], ['Log Likelihood Maximum'], ['Ground Truth']});
% set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

hAxes       = subplot(2, 1, 2);
set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vLambdaGrid, log(vLsLikelihood));
set(hLineSeries, 'LineWidth', lineWidthNormal);
hLineSeries = plot([paramLambda, paramLambda], get(hAxes, 'YLim'));
set(hLineSeries, 'LineStyle', ':');
hLineSeries = plot([paramLambdaLs, paramLambdaLs], get(hAxes, 'YLim'));
set(hLineSeries, 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Log Least Squares Objective Function Value vs. \lambda Hypothesis']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', '\lambda', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Log Least Squares Objective Function'], ['Log Least Squares Minimum'], ['Ground Truth']});
% set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

