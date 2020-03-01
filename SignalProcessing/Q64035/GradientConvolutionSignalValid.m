% Analysis of the Gradient of Convolution
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     01/03/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;
DIFF_MODE_COMPLEX   = 4;


%% Simulation Parameters

numCoefficients = 16;
numSamples      = 16;

diffMode    = DIFF_MODE_CENTRAL;
epsVal      = 1e-6;


%% Generate Data

vH = rand(numCoefficients, 1);
vX = randn(numSamples, 1);

vY = conv2(vX, vH, 'valid');

hObjFun = @(vX) 0.5 * sum((conv2(vX, vH, 'valid') - vY) .^ 2);


%% Numerical Derivative
vX0 = randn(numSamples, 1);

vDRef = CalcFunGrad(vX0, hObjFun, diffMode, epsVal);
vD = conv2((conv2(vX0, vH, 'valid') - vY), vH(end:-1:1), 'full'); %<! Works for numCoefficients <= numSamples

max(abs(vDRef - vD))


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

