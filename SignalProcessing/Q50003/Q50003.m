% StackExchange Signal Processing Q50003
% https://dsp.stackexchange.com/questions/50003/
% Finding Reference Audio Signal in Test Audio Signal and Cropping Accordingly
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     29/06/2018  Royi
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

waveFileName = 'Toto - Africa';

testSignalStartIdx  = 6001;
testSignalEndIdx    = 20000;

refSignalStartIdx   = testSignalStartIdx + 11000;
refSignalEndIdx     = refSignalStartIdx + 2000 - 1;


%% Load & Generate Data

[vA, aSamplingRate] = audioread('Toto - Africa.wav');

vTestSignal = vA(testSignalStartIdx:testSignalEndIdx);
vRefSignal  = vA(refSignalStartIdx:refSignalEndIdx);


%% Analysis

refSignalStartIdxRel   = refSignalStartIdx - testSignalStartIdx + 1;
refSignalEndIdxRel     = refSignalEndIdx - testSignalStartIdx + 1;

numSamplesTestSignal    = size(vTestSignal, 1);
numSamplesRefSignal     = size(vRefSignal, 1);

vCrossCorrelationVal            = zeros([numSamplesTestSignal - numSamplesRefSignal + 1, 1]);
vCrossCorrelationNormalizedVal  = zeros([numSamplesTestSignal - numSamplesRefSignal + 1, 1]);

vRefSignalNorm = norm(vRefSignal);

for ii = 1:size(vCrossCorrelationVal, 1)
    vTestSignalSamples                  = vTestSignal(ii:(ii + numSamplesRefSignal - 1));
    vCrossCorrelationVal(ii)            = sum(vTestSignalSamples .* vRefSignal);
    vCrossCorrelationNormalizedVal(ii)  = vCrossCorrelationVal(ii) / (vRefSignalNorm * norm(vTestSignalSamples));
end


[~, crossCorrelationMaxIdx]             = max(abs(vCrossCorrelationVal));
[~, crossCorrelationNormalizedMaxIdx]   = max(abs(vCrossCorrelationNormalizedVal));


%% Display Results

figureIdx = figureIdx + 1;

hFigure         = figure('Position', figPosLarge);
hAxes           = subplot(3, 1, 1);
set(hAxes, 'NextPlot', 'add');
hLineSeries  = line(1:numSamplesTestSignal, vTestSignal);
set(hLineSeries, 'LineWidth', lineWidthNormal);
hLineSeries  = line(refSignalStartIdxRel:refSignalEndIdxRel, vRefSignal);
set(hLineSeries, 'LineWidth', lineWidthNormal, 'Color', mColorOrder(2, :));
set(get(hAxes, 'Title'), 'String', {['Test and Reference Signals'], ['Reference Signal Start Index - ', num2str(refSignalStartIdxRel)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Test Signal'], ['Reference Signal']});

hAxes           = subplot(3, 1, 2);
hLineSeries  = line(1:(numSamplesTestSignal - numSamplesRefSignal + 1), vCrossCorrelationVal / max(abs(vCrossCorrelationVal)));
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Cross Correlation Function'], ['Cross Correlation Max Value Index - ', num2str(crossCorrelationMaxIdx)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);

hAxes           = subplot(3, 1, 3);
hLineSeries  = line(1:(numSamplesTestSignal - numSamplesRefSignal + 1), vCrossCorrelationNormalizedVal);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Normalized Cross Correlation Function'], ['Normalized Cross Correlation Max Value Index - ', num2str(crossCorrelationNormalizedMaxIdx)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);

if(generateFigures == ON)
    saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

