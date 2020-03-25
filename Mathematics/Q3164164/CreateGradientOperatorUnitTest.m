% Mathematics Q3079400
% https://math.stackexchange.com/questions/3079400
% Numerical Implementation: Solution for the Euler Lagrange Equation Of the Rudin Osher Fatemi (ROF) Total Variation Denoising Model
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     29/03/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

numRows = 5;
numCols = 4;


%% Load / Generate Data

mX = reshape(1:(numRows * numCols), numRows, numCols);


%% Gradient by MATLAB Notation

vDv = mX(1:end - 1, :) - mX(2:end, :);
vDv = vDv(:);
vDh = mX(:, 1:end - 1) - mX(:, 2:end);
vDh = vDh(:);

vDRef = [vDv; vDh];


%% Gradient by MATLAB Matrix Operation

mD = CreateGradientOperator(numRows, numCols);
vD = mD * mX(:);


%% Analysis

vE = abs(vD - vD);
max(vE)


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

