% Mathematics Q18993
% https://dsp.stackexchange.com/questions/questions/18993
% Estimate the Filter Coefficients of 1D Filtration (Convolution)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     25/09/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numSamplesX  = 150; %<! X Data Number of Samples
numCoeff    = 11; %<! Filter Number of Coefficients
numSamplesY = numSamplesX - numCoeff + 1; %<! Y Data Number of Samples

noiseStd = 0.000; %<! Noise Standard Deviation

numIterations   = 60;
stepSize        = 0.00075;


%% Generate Data

% Input Data
vX = randn([numSamplesX, 1]);

% Filter Coefficients
vH = randi([-10, 10], [numCoeff, 1]);
vH = vH - mean(vH);

% Data Samples
% Data Length = numSamples - numCoeff + 1
vY = conv2(vX, vH, 'valid') + (noiseStd * randn([numSamplesY, 1]));

hObjFun = @(vH) 0.5 * mean( (conv2(vX, vH, 'valid') - vY) .^ 2 );


%% Direct Solution

% mX      = ImageToColumnsSliding(vX, [numCoeff, 1]).'; %<! Builinding in Correlation Form
% vEstH   = flip(mX \ vY, 1); %<! Flipping as the operation is Convolution

mX      = ImageToColumnsSliding(vX, [numCoeff, 1]).'; %<! Builinding in Correlation Form
mX      = flip(mX, 2); %<! Flipping for Convolution Form
vEstH   = mX \ vY;


%%  Iterative Solution

vHEst       = zeros([numCoeff, 1]); %<! Assuming knowing its lentgh
vCostFun    = nan([numIterations, 1]);
vCostFun(1) = hObjFun(vHEst);

figureIdx = figureIdx + 1;

hFigure         = figure('Position', figPosLarge);
hAxes(1)        = subplot(2, 1, 1);
hLineSeries  = line([1:numCoeff], [vH, vHEst]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes(1), 'Title'), 'String', {['Estimateing Filter Coefficients'], ['Iteration Number - ', num2str(1)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes(1), 'XLabel'), 'String', {['Coefficient Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes(1), 'YLabel'), 'String', {['Coefficient Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['Filter Estimation']});

hAxes(2)        = subplot(2, 1, 2);
hLineSeries(3)  = line([1:numIterations], vCostFun);
set(hLineSeries(3), 'LineWidth', lineWidthNormal);
set(hAxes(2), 'XLim', [0, numIterations]);
set(hAxes(2), 'YLim', [0, max(get(hAxes(2), 'YLim'))]);
set(get(hAxes(2), 'Title'), 'String', {['Cost Function Value']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes(2), 'XLabel'), 'String', {['Iteration Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes(2), 'YLabel'), 'String', {['Cost Function Value']}, ...
    'FontSize', fontSizeAxis);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end

for ii = 2:numIterations
    vG      = conv2(flip(vX, 1), (conv2(vX, vHEst, 'valid') - vY), 'valid');
    vHEst   = vHEst - (stepSize * vG);
    
    % Analysis
    vCostFun(ii) = hObjFun(vHEst);
    
    figureIdx = figureIdx + 1;
    
    set(hLineSeries(2), 'YData', vHEst);
    set(get(hAxes(1), 'Title'), 'String', {['Estimateing Filter Coefficients'], ['Iteration Number - ', num2str(ii)]}, ...
        'FontSize', fontSizeTitle);
    set(hLineSeries(3), 'YData', vCostFun);
    drawnow();
    pause(0.1);
    
    if(generateFigures == ON)
        saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    end
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

