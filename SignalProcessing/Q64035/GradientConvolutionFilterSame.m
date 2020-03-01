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

numCoefficients = 4;
numSamples      = 12;

diffMode    = DIFF_MODE_CENTRAL;
epsVal      = 1e-6;


%% Generate Data

vH = rand(numCoefficients, 1);
vX = randn(numSamples, 1);

vY = conv2(vX, vH, 'same');

hObjFun = @(vH) 0.5 * sum((conv2(vX, vH, 'same') - vY) .^ 2);


%% Numerical Derivative
vH0 = randn(numCoefficients, 1);

vDRef = CalcFunGrad(vH0, hObjFun, diffMode, epsVal);
vDD = conv2(vX(end:-1:1), (conv2(vX, vH0, 'same') - vY), 'full');
% vD = vD((numCoefficients + 1):(numCoefficients + numCoefficients));

vD = zeros(numCoefficients, 1);
if(numCoefficients < (2 * numSamples - 1))
    firstIdx = floor(((2 * numSamples - 1) - numCoefficients) / 2) + 1;
    lastIdx = firstIdx + numCoefficients - 1;
    vD = vDD(firstIdx:lastIdx);
else
    firstIdx = ceil((numCoefficients - (2 * numSamples - 1)) / 2) + 1;
    lastIdx = firstIdx + numCoefficients - 1;
    vD(firstIdx:lastIdx) = vDD;
end

max(abs(vDRef - vD))


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

