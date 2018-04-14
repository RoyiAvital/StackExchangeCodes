% Stack Overflow Q1427602
% https://stackoverflow.com/questions/1427602
% MATLAB: Display an image in Its Original Size
% References:
%   1.  Photo: Taken by Roman Vanur (Flickr).
%       https://www.flickr.com/photos/80272075@N02/7572939538/
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     14/04/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Load Data

mI = imread('7572939538_04e373d8f4_z.jpg');

numRows = size(mI, 1);
numCols = size(mI, 2);


%% Setings

horMargin = 30;
verMargin = 60; %<! Title requires more


%% Display Image

vFigPos = [100, 100, numCols + (2 * horMargin), numRows + (2 * verMargin)]; %<! [Left, Bottom, Width, Height]
vAxesPos = [horMargin, verMargin, numCols, numRows];

hFigure = figure('Position', vFigPos, 'Units', 'pixels');
hAxes   = axes('Units', 'pixels', 'Position', vAxesPos);
hImageObj = image(hAxes, mI);
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
set(get(hAxes, 'Title'), 'String', {['Landscape by Roman Vanur']}, ...
    'Fontsize', fontSizeTitle);
set(hAxes, 'XTick', []);
set(hAxes, 'YTick', []);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

