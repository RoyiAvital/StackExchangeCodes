% StackExchange Signal Processing Q52150
% https://dsp.stackexchange.com/questions/52150
% Estimating a Signal Given a Noisy Measurement of the Signal and Its Derivative (Denoising)
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     22/04/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 7899;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Simulation Parameters

numTests    = 5000;
numSamples  = 100;
noiseStdV   = 2;
noiseStdW   = 8;
signalFreq  = 3; %<! [Hz]

paramLambdaMinVal       = 0.001;
paramLambdaMaxVal       = 0.2;
paramLambdaNumGridPts   = 200;


%% Generate Data

vT = linspace(0, 1, numSamples).';
dT = mean(diff(vT));

hGenV = @() noiseStdV * randn(numSamples, 1);
hGenW = @() noiseStdW * randn(numSamples - 1, 1); %<! Pay attention that vY is 1 sample shorter (Valid Convolution)

vX = sin(2 * pi * signalFreq * vT);

mF  = CreateConvMtx1D([1, -1], numSamples, CONVOLUTION_SHAPE_VALID) / dT;
mFF = mF.' * mF;

hGenY = @(vV) vX + vV; %<! Pay attention that vY is 1 sample shorter (Valid Convolution)
hGenZ = @(vW) (mF * vX) + vW;

paramLambda = (noiseStdV / noiseStdW) ^ 2; %<! Optimal (Analytic)
vParamLambda = linspace(paramLambdaMinVal, paramLambdaMaxVal, paramLambdaNumGridPts - 1);
vParamLambda = sort([vParamLambda, paramLambda]); %<! Inserting the analytic solution so its MSE will be evaluated correctly (Mean of many realizations)
paramLambdaIdx = find(vParamLambda == paramLambda);

mI = eye(numSamples);


%% Analysis of Lambda

mMse = zeros(paramLambdaNumGridPts, numTests);

for ii = 1:numTests
    vV = hGenV();
    vW = hGenW();
    
    vY = hGenY(vV);
    vZ = hGenZ(vW);
    
    for jj = 1:paramLambdaNumGridPts
        paramLambda = vParamLambda(jj);
        
        vXEst = (mI + (paramLambda * mFF)) \ (vY + (paramLambda * mF.' * vZ));
        mMse(jj, ii) = mean((vXEst - vX) .^ 2);
        
    end
    
end

vMse = mean(mMse, 2);
[~, paramLambdaArgMin] = min(vMse);
paramLambdaNum = vParamLambda(paramLambdaArgMin);

paramLambda = vParamLambda(paramLambdaIdx);

vXEst = (mI + (paramLambda * mFF)) \ (vY + (paramLambda * mF.' * vZ));
mseEst = vMse(paramLambdaIdx);


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = subplot(4, 1, 1);
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vT, vX);
set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Signal $ x $']}, ...
    'FontSize', fontSizeTitle, 'Interpreter', 'latex');
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);

hAxes   = subplot(4, 1, 2);
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vT, vY);
set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Signal $ y $ for $ {\sigma}_{v} = ', num2str(noiseStdV), ' $']}, ...
    'FontSize', fontSizeTitle, 'Interpreter', 'latex');
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);

hAxes   = subplot(4, 1, 3);
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vT(1:(numSamples - 1)), vZ);
set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Signal $ z $ for $ {\sigma}_{w} = ', num2str(noiseStdW), ' $']}, ...
    'FontSize', fontSizeTitle, 'Interpreter', 'latex');
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);

hAxes   = subplot(4, 1, 4);
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vT, [vX, vXEst]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Estimated Signal $ \hat{x} $']}, ...
    'FontSize', fontSizeTitle, 'Interpreter', 'latex');
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Signal $ x $'], ['Estimated Signal $ \hat{x} $']}, 'Interpreter', 'latex');

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes();
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vParamLambda, 10 * log10(vMse), paramLambda, 10 * log10(mseEst), vParamLambda(paramLambdaArgMin), 10 * log10(vMse(paramLambdaArgMin)));
set(hLineSeries(1), 'LineWidth', lineWidthNormal);
set(hLineSeries(2:3), 'LineStyle', 'none', 'Marker', 'o');
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Optimal Value of $ \lambda $']}, ...
    'FontSize', fontSizeTitle, 'Interpreter', 'latex');
set(get(hAxes, 'XLabel'), 'String', {['\lambda']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Mean Square Error [dB]']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['MSE Curve for $ \lambda $'], ['Optimal'], ['Numrical']}, 'Interpreter', 'latex');

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

