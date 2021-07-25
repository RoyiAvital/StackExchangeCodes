% StackExchange Signal Processing Q76344
% https://dsp.stackexchange.com/questions/76344
% Estimate and Track the Amplitude, Frequency and Phase of a Sine Signal Using a Kalman Filter
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     25/07/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 2132;79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

EKF_JACOBIAN_METHOD_ANALYTIC    = 1;
EKF_JACOBIAN_METHOD_NUMERIC     = 2;


%% Simulation Parameters

% Simulation parameters
numMeasurements = 1200;
dT              = 0.01;

% Model parameters
paramAmp        = 1;
paramAngFreq    = 10;
paramPhase      = 0; %<! Instant phase (paramAngFreq * T + phase)

ampNoiseVar     = 0.1;
angFreqNoiseVar = 0.2;

mF = [1, 0, 0; 0, 1, 0; 0, dT, 1];
mQ = [dT * ampNoiseVar, 0 , 0; ...
        0,  dT * angFreqNoiseVar, 0.5 * dT * dT * angFreqNoiseVar; ...
        0, 0.5 * dT * dT * angFreqNoiseVar, (dT * dT * dT * angFreqNoiseVar) / 3];
mR = 1;
mP = 3 * eye(size(mQ, 1));

hF  = @(vX) mF * vX;
hMf = @(vX) mF;
hH  = @(vX) vX(1) * sin(vX(3)); %<! Amplitude * sin(instant phase)
hMh = @(vX) [sin(vX(3)), 0, vX(1) * cos(vX(3))]; %<! Derivative of hH at vX


%% Generate Data

vX          = [paramAmp; paramAngFreq; paramPhase];
modelDim    = size(vX, 1);
measDim     = size(mR, 1);

mQChol = chol(mQ, 'lower');
mRChol = chol(mR, 'lower');
mPChol = chol(mP, 'lower');

mX          = zeros(modelDim, numMeasurements);
mX(:, 1)    = vX + (mPChol * randn(modelDim, 1));
mY          = zeros(measDim, numMeasurements);
mY(:, 1)    = hH(mX(:, 1));
mZ          = zeros(measDim, numMeasurements);
mZ(:, 1)    = hH(mX(:, 1)) + (mRChol * randn(measDim, 1));

for ii = 2:numMeasurements
    mX(:, ii) = (mF * mX(:, ii - 1)) + (mQChol * randn(modelDim, 1));
    mY(:, ii) = hH(mX(:, ii));
    mZ(:, ii) = hH(mX(:, ii)) + (mRChol * randn(measDim, 1));
end


%% Analysis

mXEst   = zeros(modelDim, numMeasurements);
tP      = zeros(modelDim, modelDim, numMeasurements);

mXEst(:, 1) = mX(:, 1) + (mPChol * randn(modelDim, 1));
tP(:, :, 1) = mP;

for ii = 2:numMeasurements
    [mXEst(:, ii), tP(:, :, ii)] = ApplyKalmanFilterIteration(mXEst(:, ii - 1), tP(:, :, ii - 1), mZ(:, ii), hF, hH, hMf, hMh, mQ, mR);
end


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes();
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(1:numMeasurements, mY);
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = plot(1:numMeasurements, mZ);
set(hLineObj, 'LineStyle', 'none', 'Marker', '*');
hLineObj = plot(1:numMeasurements, mXEst(1, :) .* sin(mXEst(3, :)));
set(hLineObj, 'LineStyle', 'none', 'Marker', 'o');
set(get(hAxes, 'Title'), 'String', {['Extended Kalman Estimation - Estimating Sine Signal Parameters']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Measurement Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['Measurement'], ['Estimation']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

