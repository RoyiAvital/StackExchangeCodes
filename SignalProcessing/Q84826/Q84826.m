% StackExchange Signal Processing Q84826
% https://dsp.stackexchange.com/questions/84826
% What Measure to Compare the Color Depth (Distribution of Colors) of Images
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     10/10/2022
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants

ENTROPY_MODE_CHANNEL    = 1; %<! Averages per channel calculation
ENTROPY_VECTOR          = 2; %<! Treats the RGB data as a single vector per pixel


%% Parameters

img001Url = 'https://i.stack.imgur.com/mYxDD.jpg';
img002Url = 'https://i.stack.imgur.com/C95vE.jpg';


%% Generate / Load Data

mI001 = imread(img001Url);
mI002 = imread(img002Url);


%% Analysis

disp(['The avergae per channel entropy of Img001 is: ', num2str(CalcImgEntropy(mI001, ENTROPY_MODE_CHANNEL))])
disp(['The avergae per channel entropy of Img002 is: ', num2str(CalcImgEntropy(mI002, ENTROPY_MODE_CHANNEL))])
disp(['The vectorized entropy of Img001 is: ', num2str(CalcImgEntropy(mI001, ENTROPY_VECTOR))])
disp(['The vectorized entropy of Img002 is: ', num2str(CalcImgEntropy(mI002, ENTROPY_VECTOR))])




%% Display Results

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


%% Auxilizary Functions




%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

