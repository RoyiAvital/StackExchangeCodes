% StackExchange Signal Processing Q76644
% https://dsp.stackexchange.com/questions/76644
% Simple and Effective Method to Estimate the Frequency of a Sine Signal in
% White Noise.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     09/08/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

% Signal Parameters
numSamples      = 200;
samplingFreq    = 10; %<! The CRLB is for Normalized Frequency

% Sine Signal Parameters
expFreq    = 1; %<! [Hz]
expAmp     = 10; %<! High value to allow high SNR
expPhase   = pi * rand();

% Analysis Parameters
numRealizations = 500;
% SNR of the Analysis (dB)
vSnrdB = linspace(-10, 40, 100).';


%% Generate Data

angFreq = 2 * pi * (expFreq / samplingFreq);

vS = expAmp * exp(1i * ((angFreq * (0:(numSamples - 1))) + expPhase));
vS = vS(:);

numNoiseStd = length(vSnrdB);
vNoiseStd   = zeros(numNoiseStd, 1);

for ii = 1:numNoiseStd
    vNoiseStd(ii) = sqrt((expAmp * expAmp) / (10 ^ (vSnrdB(ii) / 10))); 
end

tFreqErr = zeros(numRealizations, numNoiseStd, 2);


%% Analysis

for estType = 1:2
    for jj = 1:numNoiseStd
        noiseStd = vNoiseStd(jj);
        for ii = 1:numRealizations
            vW = (noiseStd / sqrt(2)) * (randn(numSamples, 1) + 1i * randn(numSamples, 1));
            vX = vS + vW;
            tFreqErr(ii, jj, estType) = expFreq - EstimateHarmonicFreqKay(vX, samplingFreq, estType);
        end
    end
end

mFreqErr = reshape(mean(tFreqErr .^ 2, 1), numNoiseStd, 2);

expMse     = expAmp * expAmp;
vNoiseVar  = vNoiseStd .^ 2;
vSnr       = expMse ./ vNoiseVar;

% See Steven Kay Estimation Theory (Pg. 57)
vFreqMseCrlb = (6 * samplingFreq * samplingFreq) ./ (((2 * pi) ^ 2) * vSnr * ((numSamples ^ 3) - numSamples));
vFreqMseCrlb = vFreqMseCrlb(:);
% In order to have any sampling rate multiply it by Fs ^ 2.


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
hLineObj = plot(vSnrdB, 10 * log10([vFreqMseCrlb, mFreqErr]));
set(hLineObj, 'LineWidth', lineWidthNormal);
% set(hLineObj(1), 'LineStyle', 'none', 'Marker', '*');
% set(hLineObj(2), 'LineStyle', 'none', 'Marker', 'x');
set(get(hAxes, 'Title'), 'String', {['MSE of Harmonic Exponential Frequency Estimation']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['SNR [dB]']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['MSE']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['CRLB'], ['Kay Estimator Type 1'], ['Kay Estimator Type 2']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

