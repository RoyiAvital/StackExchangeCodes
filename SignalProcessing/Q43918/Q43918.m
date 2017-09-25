% Mathematics Q43918
% https://dsp.stackexchange.com/questions/questions/43918
% Determine the Unit Impulse Response of ARMA Filter for 8 First Indices
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     25/09/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numSamples = 8;

% Filter Coefficients
% See MATLAB's Filter Documentation for the sign.
vA = [1; -0.5; -(1 / 8)];
vB = [0; 0; 0.5];


%% Generate Data

% Unit Impulse
vX = zeros([numSamples, 1]);
vX(1) = 1;


%% Simulate Result

vY = filter(vB, vA, vX);


%% Display Results

hFigure         = figure('Position', figPosDefault);
hAxes           = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries  = line([0:(numSamples - 1)], [vX, vY]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['ARMA Filter Unit Impulse Response']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Indices']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Unite Impulse'], ['ARMA Filter Response']});


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

