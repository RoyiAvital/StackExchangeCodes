% Stack Overflow Q15889046
% https://stackoverflow.com/questions/15889046
% Implement Contrast and Brightness Adjustment
% References:
%   1.  
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     15/09/2024
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Parameters

imgUrl      = 'https://i.postimg.cc/T1WSckLS/Lena256.png'; %<! https://i.imgur.com/HfSkNNb.png
numGridPts  = 10000;

slopeAdj    = 1.2;
brightenAdj = 0.1;


%% Load Data


mI = im2double(imread(imgUrl));

numRows = size(mI, 1);
numCols = size(mI, 2);

vX = linspace(-0.5, 0.5, numGridPts);


%% Analysis

vY = vX;
vI = mI(:) - 0.5;
vYAdj = slopeAdj * vX + brightenAdj;
vYAdj = min(max(vYAdj, -0.5), 0.5);
vIAdj = slopeAdj * vI + brightenAdj;
vIAdj = min(max(vIAdj, -0.5), 0.5);

mIAdj = reshape(vIAdj + 0.5, numRows, numCols);
mIAdj = min(max(mIAdj, 0.0), 1.0);



%% Display Data

[hFigure, hAxes, hImg] = PlotImage(mI, 'plotTitle', 'Input Image', 'openFig', 1);
[hFigure, hAxes, hImg] = PlotImage(mIAdj, 'plotTitle', 'Output Image', 'openFig', 1);



hFigure = figure('Position', figPosDefault);
hAxes   = axes(hFigure, 'Units', 'pixels');
set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vX, vY, 'DisplayName', 'Identity Mapping');
set(hLineSeries, 'LineWidth', lineWidthNormal);
hLineSeries = plot(vX, vYAdj, 'DisplayName', 'Adjusted Mapping');
set(hLineSeries, 'LineWidth', lineWidthNormal);
hLineSeries = plot(vI, vI, 'DisplayName', 'Image Data');
set(hLineSeries, 'LineStyle', 'none', 'Marker', '+');
hLineSeries = plot(vI, vIAdj, 'DisplayName', 'Image Data - Adjusted');
set(hLineSeries, 'LineStyle', 'none', 'Marker', 'o');
set(get(hAxes, 'Title'), 'String', {['Contrast and Brightness Mapping']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Input Value', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Output Value', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend();
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);







% set(hAxes, 'DataAspectRatio', [1, 1, 1]);
% set(get(hAxes, 'Title'), 'String', {['Landscape by Roman Vanur']}, ...
%     'Fontsize', fontSizeTitle);
% set(hAxes, 'XTick', []);
% set(hAxes, 'YTick', []);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

