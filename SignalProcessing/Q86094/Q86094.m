% StackExchange Signal Processing Q86094
% https://dsp.stackexchange.com/questions/86094
% Analyzing 2 2D Kernels Which Approximates a Gaussian Kernel
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     10/01/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants

KERNEL_A = 1; %<! The 1st kernel
KERNEL_B = 2; %<! The 2nd Kernel


%% Parameters

mK1 = (1 / 16) * [1, 2, 1; 2, 4, 2; 1, 2, 1];
mK2 = (1 / 48) * [0, 1, 2, 1, 0; 1, 2, 4, 2, 1; 2, 4, 8, 4, 2; 1, 2, 4, 2, 1; 0, 1, 2, 1, 0];


%% Generate / Load Data


%% Analysis

% Padding the 1st array to have the same size as the 2nd
mK1 = padarray(mK1, [1, 1], 0, 'both');

% SVD Decomposition of the kernels
[mU1, mS1, mV1] = svd(mK1);
[mU2, mS2, mV2] = svd(mK2);

% Separable (Approximation)
mD1 = mS1(1, 1) * mU1(:, 1) * mV1(:, 1).';
mD2 = mS2(1, 1) * mU2(:, 1) * mV2(:, 1).';

vU1 = sqrt(mS1(1, 1)) * mU1(:, 1);
vU2 = sqrt(mS2(1, 1)) * mU2(:, 1);

vU1U2 = vU1 - vU2;
vU2U1 = vU2 - vU1;


%% Display Results

figureIdx = figureIdx + 1;

maxVal = max(max(mK1(:)), max(mK2(:)));

hF = figure('Position', [100, 100, 900, 400]);
hA   = subplot(1, 2, 1);
hImgObj = imagesc(mK1);
set(hA, 'CLim', [0, maxVal]);
set(hA, 'DataAspectRatio', [1, 1, 1]);
set(get(hA, 'Title'), 'String', {['Kernel A']}, ...
    'FontSize', fontSizeTitle);

hA   = subplot(1, 2, 2);
hImgObj = imagesc(mK2);
set(hA, 'CLim', [0, maxVal]);
set(hA, 'DataAspectRatio', [1, 1, 1]);
set(get(hA, 'Title'), 'String', {['Kernel B']}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    % saveas(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

figureIdx = figureIdx + 1;

maxVal = max(max(mD1(:)), max(mD2(:)));

hF = figure('Position', [100, 100, 900, 400]);
hA   = subplot(1, 2, 1);
hImgObj = imagesc(mD1);
set(hA, 'CLim', [0, maxVal]);
set(hA, 'DataAspectRatio', [1, 1, 1]);
set(get(hA, 'Title'), 'String', {['Separable Approximation of Kernel A']}, ...
    'FontSize', fontSizeTitle);

hA   = subplot(1, 2, 2);
hImgObj = imagesc(mD2);
set(hA, 'CLim', [0, maxVal]);
set(hA, 'DataAspectRatio', [1, 1, 1]);
set(get(hA, 'Title'), 'String', {['Separable Approximation of Kernel B']}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    % saveas(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

figureIdx = figureIdx + 1;

hF = figure('Position', figPosLarge);
hA   = axes(hF);
set(hA, 'NextPlot', 'add');
hLineObj = plot(-mU1(:, 1), 'DisplayName', 'Kernel A');
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = plot(-mU2(:, 1), 'DisplayName', 'Kernel B');
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hA, 'Title'), 'String', {['Separable Filters of the Kernels']}, ...
    'FontSize', fontSizeTitle);
hLegend = ClickableLegend();

if(generateFigures == ON)
    % saveas(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

figureIdx = figureIdx + 1;

hF = figure('Position', [100, 100, 1200, 800]);
hA   = subplot(1, 2, 1);
[~, ~, hLineObj] = PlotDft(vU1U2, 1, 'plotTitle', 'DFT of Kernal A - Kernel B in 1D Seprable Approximation', 'numFreqBins', 100);
set(hLineObj, 'LineWidth', lineWidthNormal);

hA   = subplot(1, 2, 2);
[~, ~, hLineObj] = PlotDft(vU2U1, 1, 'plotTitle', 'DFT of Kernal B - Kernel A in 1D Seprable Approximation', 'numFreqBins', 100);
set(hLineObj, 'LineWidth', lineWidthNormal);

if(generateFigures == ON)
    % saveas(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

% figureIdx = figureIdx + 1;
% 
% hFigure = figure('Position', figPosLarge);
% hAxes   = axes(hFigure);
% set(hAxes, 'NextPlot', 'add');
% hLineObj = plot(vTheta, 10 * log10(abs(vM)));
% set(hLineObj, 'LineWidth', lineWidthNormal);
% 
% set(get(hAxes, 'Title'), 'String', {['MUSIC Pseudo Spectrum']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hAxes, 'XLabel'), 'String', {['Spatial Angle [Deg]']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hAxes, 'YLabel'), 'String', {['Value [dB]']}, ...
%     'FontSize', fontSizeAxis);
% 
% hLineObj = plot(vTheta(vPeakIdx(1:numSig)), 10 * log10(abs(vPeakVal(1:numSig))));
% set(hLineObj, 'LineStyle', 'none', 'LineWidth', lineWidthNormal, 'Marker', 'x', 'MarkerSize', markerSizeLarge, 'Color', 'r');
% for ii = 1:numSig
%     hLineObj = xline(vTheta(vPeakIdx(ii)), '-', {['Angle: ', num2str(vTheta(vPeakIdx(ii))), ' [Deg]']});
%     set(hLineObj, 'LineStyle', ':', 'LineWidth', lineWidthThin, 'Color', 'r');
% end
% 
% % hLegend = ClickableLegend({['Ground Truth'], ['Input Noisy Samples'], ['TV Estimation']});
% 
% if(generateFigures == ON)
%     % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
%     print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
% end


%% Auxiliary Functions




%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

