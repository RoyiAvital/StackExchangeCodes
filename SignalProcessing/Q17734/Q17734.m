% Signal Processing Q17734
% https://dsp.stackexchange.com/questions/17734
% Estimate the Discrete Fourier Series of a Signal with Missing Samples
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.001     17/01/2021  Royi Avital     RoyiAvital@yahoo.com
%   *   First release.
% - 1.0.000     20/07/2017
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;
run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numSamples      = 150;
numGivenSamples = 120; %<! Must be not greater than 'numSamples'
filterNumCoeff  = 15;


%% Generate Data and Samples

vFilterCoeff = rand([filterNumCoeff, 1]);
vFilterCoeff = vFilterCoeff / sum(vFilterCoeff);

% Generating the signal
vX = randn([numSamples, 1]);
vX = conv2(vX, vFilterCoeff, 'same');

% Generating the given data
vGivenIdx       = sort(randperm(numSamples, numGivenSamples));
vXX             = vX(vGivenIdx);

% Generating the DFT Matrix with the respected rows
% Pay attention to scaling in order to make it Unitary
mF = dftmtx(numSamples) / sqrt(numSamples);
mA = mF(:, vGivenIdx);


%% Estimation

% This is equivalent to padding with zeros all the samples which are not
% given.

% vY = (mA * mA') \ (mA * vXX);
vY = pinv(mA') * vXX;


%% Display Results

refDftIdx   = floor((numSamples + 1) / 2);
vDftIdx     = [0:(refDftIdx - 1)];

vXDftEstimated  = vY(1:refDftIdx);
vXDft           = mF * vX;
vXDft           = vXDft(1:refDftIdx);

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot(vDftIdx, abs([vXDft, vXDftEstimated]));
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['Estimating the DFT Given Missing Samples'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'DFT Sample Index [n]', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Amplitude', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['The DFT of the Complete Signal'], ['Estimated DFT from Partial Signal']});


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

