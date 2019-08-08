% Mathematics Q2071774
% https://math.stackexchange.com/questions/2071774
% The Proximal Operator of Cubed Euclidean Norm.
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     03/08/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 0;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;
DIFF_MODE_COMPLEX   = 4;


%% Parameters

numElements = 3;

paramLambda = 0.1;

diffMode = DIFF_MODE_COMPLEX;
epsVal = 1e-6;


%% Load / Generate Data

vY = 5 * randn(numElements, 1);


%% Projection

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numElements);
    % For 'norms()' see http://ask.cvxr.com/t/4351 and http://cvxr.com/cvx/doc/funcref.html
    minimize( (0.5 * sum_square(vX - vY)) );
    subject to
        pow_pos(norm(vX), 3) <= 1;
cvx_end

toc();

% disp([' ']);
% disp(['CVX Solution Summary']);
% disp(['The CVX Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
% disp([' ']);

vX

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numElements);
    % For 'norms()' see http://ask.cvxr.com/t/4351 and http://cvxr.com/cvx/doc/funcref.html
    minimize( (0.5 * sum_square(vX - vY)) );
    subject to
        norm(vX) <= 1;
cvx_end

toc();

% disp([' ']);
% disp(['CVX Solution Summary']);
% disp(['The CVX Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
% disp([' ']);

vX


%% The Prox Operator

% Solution by CVX

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numElements);
    % For 'norms()' see http://ask.cvxr.com/t/4351 and http://cvxr.com/cvx/doc/funcref.html
    minimize( (0.5 * sum_square(vX - vY)) + (paramLambda * pow_pos(norm(vX), 3)) );
cvx_end

toc();

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp([' ']);

vX

(1 - (paramLambda / max(norm(vY), paramLambda))) * vY

mXRef = mX;

% Analytic Solution

mX = mY .* (1 - (paramLambda ./ (  max( sqrt(sum(mY .^ 2)), paramLambda ) )));

maxAbsDev = max(abs(mX(:) - mXRef(:)));

disp(['The Maximum Absolute Deviation - ', num2str(maxAbsDev)]);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

