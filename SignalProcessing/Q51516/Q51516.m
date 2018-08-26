% Mathematics Q51516
% https://dsp.stackexchange.com/questions/51516
% FFT vs DFT Run Time Comparison (Complexity Analysis) in MATLAB
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     26/08/2018
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

vNumSamples     = 2:2:1024;
numIterations   = 6;


%% Generate Data

mDftTime = zeros(numIterations, length(vNumSamples));
mFftTime = zeros(numIterations, length(vNumSamples));

for jj = 1:length(vNumSamples)
    numSamples = vNumSamples(jj);
    vX = randn(numSamples, 1);
    
    for ii = 1:numIterations
        hDftTimer           = tic();
        vXDft               = ApplyDft(vX, numSamples);
        mDftTime(ii, jj)    = toc(hDftTimer);
        
        hFftTimer           = tic();
        vXFft               = fft(vX);
        mFftTime(ii, jj)    = toc(hFftTimer);
    end
    
end


%% Run Time Analysis

vDftMedian = median(mDftTime).';
vFftMedian = median(mFftTime).';

vDftMean = mean(mDftTime).';
vFftMean = mean(mFftTime).';

vDftMax = max(mDftTime).';
vFftMax = max(mFftTime).';

vDftMin = max(mDftTime).';
vFftMin = max(mFftTime).';


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = subplot(4, 1, 1);
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vNumSamples, [vDftMedian, vFftMedian]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['DFT vs. FFT Run Time - Median']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Input Size']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Run Time [Sec]']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['DFT'], ['FFT']});

hAxes   = subplot(4, 1, 2);
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vNumSamples, [vDftMean, vFftMean]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['DFT vs. FFT Run Time - Mean']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Input Size']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Run Time [Sec]']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['DFT'], ['FFT']});

hAxes   = subplot(4, 1, 3);
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vNumSamples, [vDftMax, vFftMax]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['DFT vs. FFT Run Time - Max']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Input Size']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Run Time [Sec]']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['DFT'], ['FFT']});

hAxes   = subplot(4, 1, 4);
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vNumSamples, [vDftMin, vFftMin]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['DFT vs. FFT Run Time - Min']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Input Size']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Run Time [Sec]']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['DFT'], ['FFT']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

