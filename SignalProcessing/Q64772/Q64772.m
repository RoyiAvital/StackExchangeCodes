% StackExchange Signal Processing Q64772
% https://dsp.stackexchange.com/questions/64772
% Detect a Frequency Change in a Step Wise Frequency Chirp
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     19/08/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

samplingFrequency   = 10000;
numSamplesSection   = 3500;
numSections         = 3;

sineAmp     = 2;
sineFreq    = 50; %<! [Hz]
sinePhase   = pi * rand(1, 1);
freqStep    = 0.1; %<! [Hz]

noiseStd    = 0.03;

% Model parameters (Tweak this for better performance)
ampNoiseVar     = 0.001;
angFreqNoiseVar = 0.0002;
msmntNoiseVar   = (noiseStd * noiseStd) / 200;


%% Generate Data

numSamples      = numSamplesSection * numSections;
numMeasurements = numSamples;

snrMeasurement = 10 * log10((sineAmp ^ 2) / (2 * noiseStd * noiseStd));

% Model
dT              = 1 / samplingFrequency;

paramAmp        = sineAmp;
paramAngFreq    = 2 * pi * sineFreq;
paramPhase      = 0; %<! Instant phase (paramAngFreq * T + phase)

mF = [1, 0, 0; 0, 1, 0; 0, dT, 1];
mQ = [dT * ampNoiseVar, 0 , 0; ...
        0,  dT * angFreqNoiseVar, 0.5 * dT * dT * angFreqNoiseVar; ...
        0, 0.5 * dT * dT * angFreqNoiseVar, (dT * dT * dT * angFreqNoiseVar) / 3];
mR = msmntNoiseVar;
mP = 3 * eye(size(mQ, 1));

hF  = @(vX) mF * vX;
hMf = @(vX) mF;
hH  = @(vX) vX(1) * sin(vX(3)); %<! Amplitude * sin(instant phase)
hMh = @(vX) [sin(vX(3)), 0, vX(1) * cos(vX(3))]; %<! Derivative of hH at vX

vX          = [paramAmp; paramAngFreq; paramPhase];
modelDim    = size(vX, 1);
measDim     = size(mR, 1);

mQChol = chol(mQ, 'lower');
mRChol = chol(mR, 'lower');
mPChol = chol(mP, 'lower');

mX          = zeros(modelDim, numMeasurements);
mX(:, 1)    = vX;
mY          = zeros(measDim, numMeasurements);
mY(:, 1)    = hH(mX(:, 1));
mZ          = zeros(measDim, numMeasurements);
mZ(:, 1)    = hH(mX(:, 1)) + (noiseStd * randn(measDim, 1));

for ii = 2:numMeasurements
    mX(:, ii) = (mF * mX(:, ii - 1));
    if(mod(ii - 1, numSamplesSection) == 0)
        mX(2, ii) = mX(2, ii) + (2 * pi * freqStep);
    end
    mY(:, ii) = hH(mX(:, ii));
    mZ(:, ii) = hH(mX(:, ii)) + (noiseStd * randn(measDim, 1));
end


%% Analysis

mXEst   = zeros(modelDim, numMeasurements);
tP      = zeros(modelDim, modelDim, numMeasurements);

mXEst(:, 1) = vX + (mPChol * randn(modelDim, 1));
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

vF      = reshape(mX(2, :) / (2 * pi), numMeasurements, 1);
vFEst   = reshape(mXEst(2, :) / (2 * pi), numMeasurements, 1);

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes();
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(1:numMeasurements, vF);
set(hLineObj, 'LineStyle', 'none', 'Marker', '*');
hLineObj = plot(1:numMeasurements, vFEst);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(hAxes, 'YLim', [sineFreq - (5 * freqStep), sineFreq + (5 * freqStep)]);
set(get(hAxes, 'Title'), 'String', {['Extended Kalman Estimation - Estimating Frequency'], ['SNR: ', num2str(snrMeasurement), ' [dB]']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Measurement Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Frequency [Hz]']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['Estimation']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

