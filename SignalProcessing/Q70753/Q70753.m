% StackExchange Signal Processing Q70753
% https://dsp.stackexchange.com/questions/70753
% Determine the Signal Curve from Parameters of a Power Curve by Noisy Measurement
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     09/10/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numRealizations = 2500;
vNoiseStd       = linspace(0, 2, 100);
numSamples      = 75;


%% Generate Data

numStd = length(vNoiseStd);
vT = linspace(0, 1, numSamples).';

mCorrectParams = false(numRealizations, numStd);

for jj = 1:numStd
    noiseStd = vNoiseStd(jj);
    for ii = 1:numRealizations
        paramAlpha  = randi(3);
        paramBeta   = randi(3);
        vX = paramAlpha * (vT .^ paramBeta);
        vY = vX + (noiseStd * randn(numSamples, 1));
        [estParamAlpha, estParamBeta] = EstimateModelParameters(vT, vY);
        
        mCorrectParams(ii, jj) = (estParamAlpha == paramAlpha) && (estParamBeta == paramBeta);
    end
end


%% Display Kernels

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100, 100, 760, 420]); %<! [x, y, width, height]
hAxes       = axes(); %<! [x, y, width, height]
hLineObj    = plot(vNoiseStd, mean(mCorrectParams));
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Parameter Estimation Success Rate as a Function of Noise STD']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Noise Std']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Success Rate [%]']}, ...
    'FontSize', fontSizeAxis);
% set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
% hLegend = ClickableLegend(cLgenedText);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

