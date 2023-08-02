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
sineAmp     = 10; %<! High value to allow high SNR

% Analysis Parameters
numRealizations = 500;
% SNR of the Analysis (dB)
vSnrdB = linspace(-10, 50, 150).';
% vSnrdB = linspace(30, 50, 150).';


%% Generate Data

% sineFreq    = 0.09078; %<! Do for [0.05, 0.10, 0.25] For no integer use 0.37.
% angFreq = 2 * pi * (sineFreq / samplingFreq);
% vS = sineAmp * sin((angFreq * (0:(numSamples - 1))) + sinePhase);
% vS = vS(:);

numNoiseStd = length(vSnrdB);
vNoiseStd   = zeros(numNoiseStd, 1);

for ii = 1:numNoiseStd
    vNoiseStd(ii) = sqrt((sineAmp * sineAmp) / (2 * 10 ^ (vSnrdB(ii) / 10))); 
end

tFreqErr = zeros(numRealizations, numNoiseStd, 2);


%% Analysis

for kk = 1:2 %<! 2 Methods
    for jj = 1:numNoiseStd
        noiseStd = vNoiseStd(jj);
        for ii = 1:numRealizations
            sineFreq = 0.05 + ((samplingFreq / 3) * rand(1, 1));
            angFreq = 2 * pi * (sineFreq / samplingFreq);
            sinePhase = 2 * pi * rand(1, 1);
            vS = sineAmp * sin((angFreq * (0:(numSamples - 1))) + sinePhase);
            vS = vS(:);
            vW = noiseStd * randn(numSamples, 1);
            vX = vS + vW;
            tFreqErr(ii, jj, 1) = sineFreq - EstimateSineFreq(vX, samplingFreq);
            tFreqErr(ii, jj, 2) = sineFreq - EstimateSineFreqCedron3Bin(vX, samplingFreq);
        end
    end
end

mFreqErr = reshape(mean(tFreqErr .^ 2, 1), numNoiseStd, 2);

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
    ['Number of Samples: ', num2str(numSamples), ', Relative Frequncy [Fc / Fs]: ', 'Random', ...
    ', Number of Realizations: ', num2str(numRealizations)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['SNR [dB]']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['MSE [dB]']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['CRLB'], ['Quadratic Model'], ['Cedron 3 Bins Model']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Auxiliary Functions

function [ estFreq ] = EstimateSineFreq( vX, samplingFreq )

numSamples = length(vX);
numSamplesDft = 2 ^ (nextpow2(numSamples) + 2);
% numSamplesDft = numSamples;

vXK = fft(vX, numSamplesDft);
vXK = abs(vXK(1:(floor(numSamplesDft / 2) + 1)));

[~, idxK] = max(vXK(3:(end - 2)));
idxK = idxK + 2;

peakShift = EstPeakShift3P(vXK((idxK - 1):(idxK + 1)));
% peakShift = EstPeakShift5P(vXK((idxK - 2):(idxK + 2)));

estFreq = (samplingFreq / numSamplesDft) * (idxK - 1 + peakShift);


end

function [ estPeakShift ] = EstPeakShift3P( vP )

vP(:) = vP - vP(2);
estPeakShift = (vP(1) - vP(3)) / (2 * (vP(1) + vP(3)));


end

function [ estPeakShift ] = EstPeakShift5P( vP )

vP(:) = vP - vP(3);
vAB = [4, -2; 1, -1; 1, 1; 4, 2] \ vP([1; 2; 4; 5]);

estPeakShift = -vAB(2) / (2 * vAB(1));


end

function [ estFreq ] = EstimateSineFreqCedron( vX, samplingFreq )
% Exact Frequency Formula for a Pure Real Tone
% Cedron Dawg
% https://www.dsprelated.com/showarticle/773.php

numSamples = length(vX);

vXK = fft(vX); %<! MATLAB doesn't have real fft
vXK = vXK(1:(floor(numSamples / 2) + 1));

[~, idxK] = max(abs(vXK(2:(end - 1))));
idxK = idxK + 1;

vP = vXK((idxK - 1):(idxK + 1));

r = exp(-1j * 2 * pi / numSamples);
vCosB = cos(2 * pi / numSamples * [idxK - 2; idxK - 1; idxK]); %<! Zero based like DFT
num = (-vP(1) * vCosB(1)) + (vP(2) * (1 + r) * vCosB(2)) - (vP(3) * r * vCosB(3));
den = -vP(1) + (vP(2) * (1 + r)) - (vP(3) * r);

estFreq = real(acos(num / den)) / (2 * pi) * samplingFreq;


end

function [ estFreq ] = EstimateSineFreqCedron3Bin( vX, samplingFreq )
% Improved Three Bin Exact Frequency Formula for a Pure Real Tone in a DFT
% Cedron Dawg
% https://www.dsprelated.com/showarticle/1108.php

numSamples = length(vX);

vXK = fft(vX); %<! MATLAB doesn't have real fft
vXK = vXK(1:(floor(numSamples / 2) + 1));

[~, idxK] = max(abs(vXK(2:(end - 1))));
idxK = idxK + 1;

vZ = vXK((idxK - 1):(idxK + 1));

vR = real(vZ);
vI = imag(vZ);
iRoot2 = 1 / sqrt(2);
vBetas = [idxK - 2; idxK - 1; idxK] * (2 * pi / numSamples); %<! Zero based like DFT
vCosB = cos(vBetas);
vSinB = sin(vBetas);

vA = [iRoot2 * (vR(2) - vR(1)); iRoot2 * (vR(2) - vR(3)); vI(1); vI(2); vI(3)];
vB = [iRoot2 * (vCosB(2) * vR(2) - vCosB(1) * vR(1)); iRoot2 * (vCosB(2) * vR(2) - vCosB(3) * vR(3)); vCosB(1) * vI(1); vCosB(2) * vI(2); vCosB(3) * vI(3)];
vC = [iRoot2 * (vCosB(2) - vCosB(1)); iRoot2 * (vCosB(2) - vCosB(3)); vSinB(1); vSinB(2); vSinB(3)];

vP = vC / norm(vC);
vD = vA + vB;
vK = vD - (vD' * vP) * vP;
num = vK' * vB;
den = vK' * vA;

estFreq = acos(max(min(num / den, 1), -1)) / (2 * pi) * samplingFreq;


end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

