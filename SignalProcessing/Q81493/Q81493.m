% StackExchange Signal Processing Q81493
% https://dsp.stackexchange.com/questions/81493
% Applying 2D Sinc Interpolation for Upsampling in the Fourier Domain (DFT / FFT)
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     29/12/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numRowsI = 5000;
numColsI = 5200;

numRowsO = 10000;
numColsO = 10400;

sincRadius = 5;


%% Generate / Load Data

mX      = GenTest([numRowsI, numColsI], sincRadius);
mYRef   = GenTest([numRowsO, numColsO], sincRadius);

mY = DftReSample2D(mX, [numRowsO, numColsO]);


%% Analysis

disp(['The interpolation error is given by: ', num2str(max(abs(mYRef - mY), [], 'all'))]);


%% Auxiliary Function

function [ mX ] = GenTest( vSize, sincRadius )

vX = linspace(-sincRadius, sincRadius, vSize(2) + 1);
vX(end) = [];
vY = linspace(-sincRadius, sincRadius, vSize(1) + 1);
vY = vY(:);
vY(end) = [];

% mX = abs(vX) + abs(vY) + sinc(sqrt(vX .^2 + vY .^2));
mX = sinc(sqrt(vX .^2 + vY .^2));

end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

