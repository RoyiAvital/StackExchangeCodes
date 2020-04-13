% Mathematics Q3619669
% https://math.stackexchange.com/questions/3619669
% Variation of Least Squares with Symmetric Positive Semi Definite (PSD)
% Constraint
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     12/04/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 0;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;
DIFF_MODE_COMPLEX   = 4;


%% Parameters

numRows = 3;
numCols = 4;

diffMode    = DIFF_MODE_CENTRAL;
epsVal      = 1e-6;

numElements = 2; %<! Works for 1. Probably the Numerical Differentiation doesn't work for more than that.
numElements = min(numElements, min(numRows, numCols));


%% Load / Generate Data

mX = randn(numRows, numCols);
% mX = eye(numRows, numCols);
[mUU, mSS, mVV] = svd(mX);

vS = diag(mSS);
maxVal = max(vS);
vS(1:numElements) = ceil(maxVal);
mSS = SetDiag(mSS, vS);
mX = mUU * mSS * mVV.';

% hObjFun = @(vX) max(svd(reshape(vX, numRows, numCols)));
% hObjFun = @(vX) norm(reshape(vX, numRows, numCols), 2);
hObjFun = @(vX) sqrt(max(eig(reshape(vX, numRows, numCols).' * reshape(vX, numRows, numCols)))); %<! Seems to be analytic but not continious


%% Numerical Solution

mGRef = reshape(CalcFunGrad(mX(:), hObjFun, diffMode, epsVal), numRows, numCols);


%% Analytic Solution

[mU, mS, mV] = svd(mX);

mG = zeros(numRows, numCols);
for ii = 1:numElements
    mG = mG + (mU(:, ii) * mV(:, ii).');
end

mG = (mU(:, 1:numElements) * mV(:, 1:numElements).');

mG ./ mGRef


%% Display Results

abs(max(mGRef(:) - mG(:)))


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

