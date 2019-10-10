% StackExchange Signal Processing Q32137
% https://dsp.stackexchange.com/questions/32137
% Frequency Analysis of a Signal without Constant Sampling Frequency
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     10/10/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

samplingFrequency = 101; %<! [Hz]
samplingInterval = 1 / samplingFrequency; %<! [Sec]
startTime = 1; %<! [Sec]
endTime = 4; %<! [Sec]
timeInterval = endTime - startTime; %<! [Sec]

numSamples = round(samplingFrequency * timeInterval); %<! Also the assumed number of samples in Frequency Domain (It doesn't have to be)
numSamplesTT = round(1.2 * numSamples);

signalFreq = 2; %!< [Hz]

% The uniform time grid
vT      = linspace(startTime, endTime, numSamples + 1);
vT(end) = [];
vT      = vT(:);

% The non uniform time grid - Reconstruction
vTT = endTime * rand(numSamplesTT, 1);
vTT = sort(vTT, 'ascend');

% The non uniform time grid - DFT
vTD = linspace(startTime, endTime, (10 * numSamples) + 1);
vTD(end) = [];
vTD = vTD(sort(randperm(length(vTD), numSamples)));
vTD = vTD(:);

% The uniform frequency grid
vF      = (samplingFrequency / 2) * linspace(-1, 1, numSamples + 1);
vF(end) = [];
vF      = vF(:);

vK = [-floor(numSamples / 2):floor((numSamples - 1) / 2)];
vK = vK(:);


%% Generate Data

vX  = cos(2 * pi * signalFreq * vT);
vFx = fftshift(fft(vX));


figureIdx = figureIdx + 1;

hFigure         = figure('Position', figPosLarge);
hAxes           = subplot(1, 2, 1);
hLineSeries     = plot(vT, vX);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Reference Signal']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Time Index']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeTitle);

hAxes           = subplot(1, 2, 2);
hStemObj = stem(vF, abs(vFx));
set(hStemObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['DFT of the Reference Signal']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Frequency [Hz]']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'YLabel'), 'String', {['Magnitude']}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Analysis - Reconstruction

mD = exp(1j * 2 * pi * (vTT / timeInterval) * vK.') / numSamples;

% Reconstruction according to the model
vY = real(mD * vFx);

figureIdx = figureIdx + 1;

hFigure         = figure('Position', figPosLarge);
hAxes           = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries     = plot(vT, vX);
set(hLineSeries, 'LineWidth', lineWidthNormal);
hLineSeries     = plot(vTT, vY);
set(hLineSeries, 'LineWidth', lineWidthNormal, 'LineStyle', ':', 'Marker', '*');
set(get(hAxes, 'Title'), 'String', {['Uniform Signal & Non Uniform Signal']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Time Index']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeTitle);
hLegend = ClickableLegend({['Uniform Signal'], ['Non Uniform Signal']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Analysis - DFT of the Non Uniformly Sampled Data

vY  = cos(2 * pi * signalFreq * vTD);

mD = exp(1j * 2 * pi * (vTD / timeInterval) * vK.') / numSamples;
vFy = pinv(mD) * vY;

figureIdx = figureIdx + 1;

hFigure         = figure('Position', figPosLarge);
hAxes           = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries     = plot(vT, vX);
set(hLineSeries, 'LineWidth', lineWidthNormal);
hLineSeries     = plot(vTD, vY);
set(hLineSeries, 'LineWidth', lineWidthNormal, 'LineStyle', ':', 'Marker', '*');
set(get(hAxes, 'Title'), 'String', {['Uniform Signal & Non Uniform Signal']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Time Index']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeTitle);
hLegend = ClickableLegend({['Uniform Signal'], ['Non Uniform Signal']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
set(hAxes, 'NextPlot', 'add');
hStemObj    = stem(vF, abs([vFx, vFy]));
set(hStemObj, 'LineWidth', lineWidthNormal);
% hLineSeries     = plot(vTT, vY);
% set(hLineSeries, 'LineWidth', lineWidthNormal, 'LineStyle', ':', 'Marker', '*');
set(get(hAxes, 'Title'), 'String', {['DFT of the Uniform Signal & Non Uniform Signal']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Frequency [Hz]']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'YLabel'), 'String', {['Magnitude']}, ...
    'FontSize', fontSizeTitle);
hLegend = ClickableLegend({['Uniform Signal'], ['Non Uniform Signal']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

