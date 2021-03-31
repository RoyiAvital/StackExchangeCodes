% StackExchange Signal Processing Q74024
% https://dsp.stackexchange.com/questions/74024
% Estimation of Amplitude, Frequency and Phase of Linear Combination of Harmonic Signal Beyond the Leakage Resolution of DFT
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     31/03/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numSignals      = 4;
samplingFreq    = 50; %<! [Hz]
timeInterval    = 12; %<! [Sec]

noiseStd = 0.0005;

% TV
paramLambda     = 0.0025;
numIterations   = 350;
noiseStdFactor  = 100;


%% Generate Data

numSamples  = (timeInterval * samplingFreq); %<! See https://dsp.stackexchange.com/a/72595
vT          = linspace(0, timeInterval, numSamples + 1); %<! See https://dsp.stackexchange.com/a/72595
vT(end)     = []; %<! See https://dsp.stackexchange.com/a/72595
vT          = vT(:);

vA = rand(numSignals, 1); %<! Amplitude
vF = [0.95; 1.00; 1.05; 1.10]; %<! Freqeuncy [Hz]
vP = pi * rand(numSignals, 1); %<! Phase

vX = zeros(numSamples, 1);
for ii = 1:numSignals
    vX = vX + (vA(ii) * sin((2 * pi * vF(ii) * vT) + vP(ii)));
end

vY = vX + (noiseStd * randn(numSamples, 1));


%% Display the Signal and DFT

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100, 100, 760, 420]); %<! [x, y, width, height]
hAxes       = axes(); %<! [x, y, width, height]
hLineObj    = plot(vT, [vX, vY]);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Data Signal - Linear Combination of Harmonic Signals'], ['Number of Signals - ', num2str(numSignals)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Time [Sec]']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Amplitude']}, ...
    'FontSize', fontSizeAxis);
% set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
hLegend = ClickableLegend({['Noiseless Data'], ['Noise STD - ', num2str(noiseStd)]});

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

figureIdx = figureIdx + 1;

[hFigure, hAxes, hLineObj] = PlotDft([vX, vY], samplingFreq, 'openFig', ON, 'plotLegend', {['Noiseless Data'], ['Noise STD - ', num2str(noiseStd)]}, 'plotLegendFlag', ON);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['The DFT of the Data Signal'], ['Number of Signals - ', num2str(numSignals)]}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Estimating the Frequencies

% Estimated Complex Exponent number of signals. For real signals it will
% ouput 2M signals as it will output negative and positive frequencies.
numSignalsEst = EstimateHarmonicSignalOrder(vX, 8 * numSignals, ceil(numSamples / 2)) / 2;

% Designed for Complex Exponent. Hence for signal it will generate +-f.
% hence we need 2 * numSaignals and filter only the positive ones.
% EstimateSinesNum() generetes 2M as model order for real signal.
vFEst = EstimateHarmonicSignalFreq(vY, 2 * numSignalsEst, ceil(numSamples / 2), samplingFreq);
vFEst = vFEst(vFEst > 0);


%% Estimating the Amplitude and Phase (Non Linear Least Squares)

hF = @(vTheta) LeastSquaresAmplitudePhase(vTheta, vFEst, vT, vY);

vThetaInit = [ones(numSignalsEst, 1); zeros(numSignalsEst, 1)];

vL = zeros(numSignalsEst, 1);
vU = [10 * ones(numSignalsEst, 1); pi * ones(numSignalsEst, 1)];
sOpt = optimoptions(@lsqnonlin, 'Algorithm', 'trust-region-reflective', 'FiniteDifferenceType', 'central');

vThetaEst = lsqnonlin(hF, vThetaInit, vL, vU, sOpt);


%% Estimating the Model Parameters (Non Linear Least Squares)

hF = @(vTheta) LeastSquaresAmplitudeFreqPhase(vTheta, vT, vY);

vThetaInit = [vThetaEst(1:(numSignalsEst)); vFEst; vThetaEst((numSignalsEst + 1):end)];

vL = zeros(size(vThetaInit, 1), 1);
vU = [10 * ones(2 * numSignalsEst, 1); pi * ones(numSignalsEst, 1)];
sOpt = optimoptions(@lsqnonlin, 'Algorithm', 'trust-region-reflective', 'FiniteDifferenceType', 'central');

vThetaEst = lsqnonlin(hF, vThetaInit, vL, vU, sOpt);
[vA; vF; vP]


%% Display Estimated Signal

vZ = zeros(numSamples, 1);

for ii = 1:numSignals
    vZ = vZ + (vThetaEst(ii) * sin((2 * pi * vThetaEst(ii + numSignals) * vT) + vThetaEst(ii + numSignals + numSignals)));
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100, 100, 760, 420]); %<! [x, y, width, height]
hAxes       = axes(); %<! [x, y, width, height]
hLineObj    = plot(vT, [vX, vZ]);
set(hLineObj(1), 'LineWidth', lineWidthNormal);
set(hLineObj(2), 'LineStyle', 'none', 'Marker', '*');
set(get(hAxes, 'Title'), 'String', {['Model Signal vS Estimated Signal'], ['Number of Signals - ', num2str(numSignals)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Time [Sec]']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Amplitude']}, ...
    'FontSize', fontSizeAxis);
% set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
hLegend = ClickableLegend({['Ground Truth'], ['Estimated Signal']});

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Cost Functions

function [ vR ] = LeastSquaresAmplitudePhase( vTheta, vFEst, vT, vY )

numSignals = size(vFEst, 1);
numSamples = size(vY, 1);

vX = zeros(numSamples, 1);

for ii = 1:numSignals
    vX = vX + (vTheta(ii) * sin((2 * pi * vFEst(ii) * vT) + vTheta(ii + numSignals)));
end

vR = vX - vY;


end


function [ vR ] = LeastSquaresAmplitudeFreqPhase( vTheta, vT, vY )

numSignals = size(vTheta, 1) / 3;
numSamples = size(vY, 1);

vX = zeros(numSamples, 1);

for ii = 1:numSignals
    vX = vX + (vTheta(ii) * sin((2 * pi * vTheta(ii + numSignals) * vT) + vTheta(ii + numSignals + numSignals)));
end

vR = vX - vY;


end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

