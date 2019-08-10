% Mathematics Q2071774
% https://math.stackexchange.com/questions/2071774
% The Proximal Operator of Cubed Euclidean Norm (Proximal Operator of Norm Composition - Cubic Euclidean Norm).
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

subStreamNumberDefault = 123;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Parameters

numElements = 50;
paramLambda = 0.25;


%% Load / Generate Data

vY = 5 * randn(numElements, 1);


%% CVX Solution

tic();

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numElements);
    minimize( (0.5 * sum_square(vX - vY)) + (paramLambda * pow_pos(norm(vX), 3)) );
cvx_end

toc();

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp([' ']);

vXRef = vX;


%% The Prox Operator - Analytic Solution

vX = (2 * vY) ./ (1 + sqrt(1 + (12 * paramLambda * norm(vY))));

%% Analysis

vE = vX - vXRef;
maxAbsDev = max(abs(vE));

disp(['The Maximum Absolute Deviation - ', num2str(maxAbsDev)]);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

