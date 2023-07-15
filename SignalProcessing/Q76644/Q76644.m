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
numSamples      = 100;
samplingFreq    = 1; %<! The CRLB is for Normalized Frequency

% Sine Signal Parameters (Non integeres divsiors of N requires much more realizations).
sineFreq    = 0.25; %<! Do for [0.05, 0.10, 0.25] For no integer use 0.37.
sineAmp     = 10; %<! High value to allow high SNR

% Analysis Parameters
numRealizations = 50;
% SNR of the Analysis (dB)
vSnrdB = linspace(-10, 50, 150).';


%% Generate Data

angFreq = 2 * pi * (sineFreq / samplingFreq);
% vS = sineAmp * sin((angFreq * (0:(numSamples - 1))) + sinePhase);
% vS = vS(:);

numNoiseStd = length(vSnrdB);
vNoiseStd   = zeros(numNoiseStd, 1);

for ii = 1:numNoiseStd
    vNoiseStd(ii) = sqrt((sineAmp * sineAmp) / (2 * 10 ^ (vSnrdB(ii) / 10))); 
end

tFreqErr = zeros(numRealizations, numNoiseStd, 2);


%% Analysis

for estType = 1:2
    for jj = 1:numNoiseStd
        noiseStd = vNoiseStd(jj);
        for ii = 1:numRealizations
%             sineFreq = (samplingFreq / 20) * rand(1, 1);
%             angFreq = 2 * pi * (sineFreq / samplingFreq);
            sinePhase = 2 * pi * rand(1, 1);
            vS = sineAmp * sin((angFreq * (0:(numSamples - 1))) + sinePhase);
            vS = vS(:);
            vW = noiseStd * randn(numSamples, 1);
            vX = vS + vW;
            % tFreqErr(ii, jj, estType) = sineFreq - EstimateSineFreqKay(vX, samplingFreq, estType);
            % It seems to be biased estimator for Sine / Cosine. With
            % Hilbert Transfrom it makes it work just as in theory.
            tFreqErr(ii, jj, estType) = sineFreq - EstimateHarmonicFreqKay(hilbert(vX), samplingFreq, estType);
        end
    end
end

mFreqErr = reshape(mean(tFreqErr .^ 2, 1), numNoiseStd, 2);
% for ii = 1:size(mFreqErr, 2)
%     mFreqErr(:, ii) = smooth(mFreqErr(:, ii), 5, 'moving');
% end

sineMse     = (sineAmp * sineAmp) / 2;
vNoiseVar   = vNoiseStd .^ 2;
vSnr        = sineMse ./ vNoiseVar;

% See Steven Kay Estimation Theory (Pg. 57)
vFreqMseCrlb = (12 * samplingFreq * samplingFreq) ./ (((2 * pi) ^ 2) * vSnr * ((numSamples ^ 3) - numSamples));
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
set(hAxes, 'YLim', [-120, 0]);
set(get(hAxes, 'Title'), 'String', {['MSE of Sine Frequency Estimation'], ...
    ['Number of Samples: ', num2str(numSamples), ', Relative Frequncy [Fc / Fs]: ', num2str(sineFreq / samplingFreq), ...
    ', Number of Realizations: ', num2str(numRealizations)]}, ...
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

