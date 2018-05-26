% StackExchange Signal Processing Q49121
% https://dsp.stackexchange.com/questions/49121/
% Finding the Best Gaussian Smoothing Kernel to Minimize the Discrepancy Between Two Images
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     11/05/2018  Royi
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;

STD_TO_RADIUS_FACTOR = 5;


%% Simulation Parameters

numRows = 200;
numCols = 200;
numExp  = 50;


kernelStdUpperBound = 25;
kernelStdLowerBound = 0.001;

sSolverOptions = optimoptions('lsqnonlin', 'OptimalityTolerance', 1e-9, ...
    'FunctionTolerance', 1e-9, 'StepTolerance', 1e-9, 'FiniteDifferenceType', 'central');


%% Analysis

numPx           = numRows * numCols;
mEstResults     = zeros([numExp, 2]); %<! mEstResults(i, 1) = Reference, mEstResults(i, 2) = Estimation
initKernelStd   = kernelStdUpperBound;

for ii = 1:numExp
    mA = rand([numRows, numCols]);
    
    gaussianKernelStd = kernelStdLowerBound + ((kernelStdUpperBound - kernelStdLowerBound) * rand(1));
    mB = ApplyGaussianBlur(mA, gaussianKernelStd, STD_TO_RADIUS_FACTOR);
    
    % Objective Functions
    hObjFun = @(kernelStd) reshape(ApplyGaussianBlur(mA, kernelStd, STD_TO_RADIUS_FACTOR) - mB, [numPx, 1]);
    estKernelEst    = lsqnonlin(hObjFun, initKernelStd, kernelStdLowerBound, kernelStdUpperBound, sSolverOptions);
    
    mEstResults(ii, 1) = gaussianKernelStd;
    mEstResults(ii, 2) = estKernelEst;
end

estRmse = sqrt(mean((mEstResults(:, 1) - mEstResults(:, 2)) .^ 2));


%% Display Results

figureIdx = figureIdx + 1;

hFigure         = figure('Position', figPosLarge);
hAxes           = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries  = line(1:numExp, mEstResults);
set(hLineSeries, 'LineStyle', 'none');
set(hLineSeries(1), 'Marker', 'o', 'MarkerSize', markerSizeLarge);
set(hLineSeries(2), 'Marker', '*', 'MarkerSize', markerSizeLarge);
set(get(hAxes, 'Title'), 'String', {['Gaussian Blur Parameter Estimation'], ['RMSE - ', num2str(estRmse)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Experiment Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Kernel STD Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['Estimated']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

