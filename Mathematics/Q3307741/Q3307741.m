% Mathematics Q3307741
% https://math.stackexchange.com/questions/3307741
% The Sub Gradient and the Prox Operator of the of L2,1 Norm (Mixed Norm)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     31/07/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;
DIFF_MODE_COMPLEX   = 4;


%% Parameters

numRows = 2;
numCols = 3;

numRowsA = 4;
numColsA = numRows;

diffMode = DIFF_MODE_COMPLEX;
epsVal = 1e-6;


%% Load / Generate Data

mX = randn(numRows, numCols);
% mX(:, 1) = 0;
mY = randn(numRows, numCols);
mA = randn(numRowsA, numColsA);
% mA = eye(numColsA);
% vOnes = ones(numRows, 1);

hMixedNormL21 = @(vX) sum(sqrt(sum((mA * reshape(vX, numRows, numCols)) .^ 2, 1)));

paramLambda = 0.1;


%% The Sub Gradient

% Numerical Gradient
vG = CalcFunGrad(mX(:), hMixedNormL21, diffMode, epsVal);

% From https://math.stackexchange.com/questions/2035198
% Only for the case mA is identity matrix
% mX ./ sqrt(vOnes.' * (mX .* mX))

vD = sqrt(sum( (mA * mX) .^ 2 ));
vDD = 1 ./ vD;
vDD(vD == 0) = 0;
mD = diag(vDD);
mG = mA.' * mA * mX * mD;

maxAbsDev = max(abs(vG - mG(:)));

disp(['The Maximum Absolute Deviation - ', num2str(maxAbsDev)]);



%% The Prox Operator

% Solution by CVX

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable mX(numRows, numCols);
    % For 'norms()' see http://ask.cvxr.com/t/4351 and http://cvxr.com/cvx/doc/funcref.html
    minimize( (0.5 * sum_square(mX(:) - mY(:))) + (paramLambda * sum(norms(mX, 2, 1))) );
cvx_end

toc();

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp([' ']);

mXRef = mX;

% Analytic Solution

mX = mY .* (1 - (paramLambda ./ (  max( sqrt(sum(mY .^ 2)), paramLambda ) )));

maxAbsDev = max(abs(mX(:) - mXRef(:)));

disp(['The Maximum Absolute Deviation - ', num2str(maxAbsDev)]);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

