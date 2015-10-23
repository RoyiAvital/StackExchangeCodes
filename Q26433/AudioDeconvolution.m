% Audio Deconvolution
% See 

%% General Parameters and Initialization

% The follwoing code must be executed before any of the following cells.
% The cells must be executed one after one in the order they are written.
clear();
close('all');

set(0, 'DefaultFigureWindowStyle', 'docked');
defaultLooseInset = get(0, 'DefaultAxesLooseInset');
set(0, 'DefaultAxesLooseInset', [0.05, 0.05, 0.05, 0.05]);

titleFontSize   = 14;
axisFotnSize    = 12;
stringFontSize  = 12;

thinLineWidth   = 2;
normalLineWidth = 3;
thickLineWidth  = 4;

smallSizeData   = 36;
mediumSizeData  = 48;
bigSizeData     = 60;

randomNumberStream = RandStream('mlfg6331_64', 'NormalTransform', 'Ziggurat');
subStreamNumber = 57162;
set(randomNumberStream, 'Substream', subStreamNumber);
RandStream.setGlobalStream(randomNumberStream);

addpath(genpath('RawData'));
addpath(genpath('AuxiliaryFunctions'));

%% Loading Data

load('input.mat');
load('handel.mat');

vInputSignal    = input;
vRefSignal      = handel;

vBlurKernel     = [0.0545; 0.2442; 0.4026; 0.2442; 0.0545];
vGradientKernel = [-1; 1]; % Forward Finite Differences
% vGradientKernel = [1; -1]; % Backward Finite Differences

%% Parameters
% Limiting the Number of Samples due to Memory Constraints
numSamplesToProcess = 800;

numSamples = size(vInputSignal, 1);

numSamples = min([numSamplesToProcess, numSamples]);

vInputSignal    = vInputSignal(1:numSamples);
vRefSignal      = vRefSignal(1:numSamples);

% Creating the Operators in Matrix Form
mBlurKernel     = convmtx(vBlurKernel, numSamples);
mGradientKernel = convmtx(vGradientKernel, numSamples);

% Enforcing size to match Convolution using 'same' property
mBlurKernel     = mBlurKernel(1:numSamples, :);
mGradientKernel = mGradientKernel(1:numSamples, :);

regFactor = 0.02; % Regularization Factor - Lambda

%% Least Squares Solution

vLsRestoredSignal = ((mBlurKernel.' * mBlurKernel) \ mBlurKernel.') * vInputSignal;

%% Regularized (Tikhonov) Least Squares Solution

vRegLsRestoredSignal = (((mBlurKernel.' * mBlurKernel) + (regFactor * (mGradientKernel.' * mGradientKernel)))\ mBlurKernel.') * vInputSignal;

%% Display Results

hFigure = figure();
hAxes   = axes();
hLineSeries = plot(1:numSamples, [vRefSignal, vInputSignal]);
set(hLineSeries, 'LineWidth', normalLineWidth);
set(get(hAxes, 'Title'), 'String', ['Reference and Input Signal'], ...
    'FontSize', titleFontSize);
set(get(hAxes, 'XLabel'), 'String', 'Samples Index', ...
    'FontSize', axisFotnSize);
set(get(hAxes, 'YLabel'), 'String', 'Samples Value', ...
    'FontSize', axisFotnSize);
hLegend = legend({['Reference Signal'], ['Input Signal']});


hFigure = figure();
hAxes   = axes();
hLineSeries = plot(1:numSamples, [vRefSignal, vLsRestoredSignal, vRegLsRestoredSignal]);
set(hLineSeries, 'LineWidth', normalLineWidth);
set(get(hAxes, 'Title'), 'String', ['Restored Signa'], ...
    'FontSize', titleFontSize);
set(get(hAxes, 'XLabel'), 'String', 'Samples Index', ...
    'FontSize', axisFotnSize);
set(get(hAxes, 'YLabel'), 'String', 'Samples Value', ...
    'FontSize', axisFotnSize);
hLegend = legend({['Reference Signal'], ['Least Squares Restored Signal'], ['Regularized Least Squares Restored Signal']});



%% Restore Defaults
set(0, 'DefaultFigureWindowStyle', 'normal');
set(0, 'DefaultAxesLooseInset', defaultLooseInset);

