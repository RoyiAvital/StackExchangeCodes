% StackExchange Signal Processing Q44115
% https://dsp.stackexchange.com/questions/questions/44115
% Show That the Power Spectrum Density Matrix Is Positive Semi Definite (PSD) Matrix
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     03/10/2017  Royi
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

dimOrder    = 4;
numSamples = 512;


%% Generate Data

mX = randn([dimOrder, numSamples]);


%% Calculate the Auto Covariance (Matrix) Function and Power Spectral Density Matrix

[mAutoCovarianceMatrix, vCorrLags]  = xcorr(mX.', 'coeff'); %<! Matrix elements are vectorized
[mAutoCovarianceMatrix, vCorrLags]  = xcorr(mX.', 'biased'); %<! Matrix elements are vectorized
mPowerSpectralDensity               = fft(ifftshift(mAutoCovarianceMatrix, 1)); %<! Making it look Symmetrical as it is
% mPowerSpectralDensity               = fft([mAutoCovarianceMatrix; mAutoCovarianceMatrix], [], 1);

% reshape(mPowerSpectralDensity(1023, :), [dimOrder, dimOrder])
% eig(reshape(mPowerSpectralDensity(1023, :), [dimOrder, dimOrder]))

mPsdEig = zeros([size(mAutoCovarianceMatrix, 1), dimOrder]);

for ii = 1:size(mAutoCovarianceMatrix, 1)
    mPsdEig(ii, :) = real(eigs(reshape(mPowerSpectralDensity(ii, :), [dimOrder, dimOrder]))).';
end

cLegendString = cell([1, dimOrder]);

for ii = 1:dimOrder
    cLegendString{ii} = ['Eigen value #00', num2str(ii)];
end


%% Display Results

hFigure         = figure('Position', figPosDefault);
hAxes           = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries  = line(vCorrLags, mPsdEig);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Eigen Values']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Lag Indices']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Eigen Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend(cLegendString);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

