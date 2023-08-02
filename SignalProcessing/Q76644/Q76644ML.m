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
numRealizations = 10;
% SNR of the Analysis (dB)
% vSnrdB = linspace(-10, 50, 150).';
vSnrdB = linspace(30, 50, 150).';

hGenSineModel = @(vParams, vN) GenSineModel(vN, vParams, samplingFreq);
vTheta = [1; 0.04; 0];
vL = [0; 0; 0];
vU = [inf; 0.5; 2 * pi];
sOptParams = optimoptions('lsqcurvefit', 'Diagnostics', 'off', 'Display', 'iter', 'FunctionTolerance', 1e-7);


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

mFreqErr = zeros(numRealizations, numNoiseStd);


%% Analysis
vN = (0:(numSamples - 1));
vN = vN(:);

for jj = 1:numNoiseStd
    noiseStd = vNoiseStd(jj);
    for ii = 1:numRealizations
        sineFreq = 0.05 + ((samplingFreq / 3) * rand(1, 1));
        sineFreq = 0.0452;
        angFreq = 2 * pi * (sineFreq / samplingFreq);
        sinePhase = 2 * pi * rand(1, 1);
        vS = sineAmp * sin((angFreq * vN) + sinePhase);
        vW = noiseStd * randn(numSamples, 1);
        vX = vS + vW;
        vParams = lsqcurvefit(hGenSineModel, vTheta, vN, vS, vL, vU, sOptParams);
        mFreqErr(ii, jj) = sineFreq - vParams(2);
    end
end

vE = zeros(1000, 1);
vG = linspace(0.01, 0.45, 1000);
for ii = 1:1000
    vSS = 10 * sin(((2 * pi * (vG(ii) / samplingFreq)) * vN) + 0);
    vE(ii) = mean((vSS - vS) .^ 2);
end




vFreqErr = mean(mFreqErr .^ 2, 1);
vFreqErr = vFreqErr(:);

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
hLineObj = plot(vSnrdB, 10 * log10([vFreqMseCrlb, vFreqErr]));
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

function [ vS ] = GenSineModel( vN, vParams, samplingFreq )


sineAmp = vParams(1);
sineFreq = vParams(2);
sinePhase = vParams(3);
angFreq = 2 * pi * (sineFreq / samplingFreq);
vS = sineAmp * sin((angFreq * vN) + sinePhase);
% vS = 10 * sin((angFreq * vN) + 0);

end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

