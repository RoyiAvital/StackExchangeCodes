% Mathematics Q2791227
% https://math.stackexchange.com/questions/2301266
% Proximal Operator of Huber Loss Function (For L1 Regularized Huber Loss)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     20/03/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 0; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

numElements = 25;

outOfSetThr     = 1e-6;
outOfSetCost    = 1e9;

paramLambda = rand(1);
paramDelta = 3;


%% Load / Generate Data

vY      = 2 * randn(numElements, 1);
hObjFun = @(vX) 0.5 * sum((vX - vY) .^ 2) + (paramLambda * HuberLoss(vX, paramDelta));
% From https://math.stackexchange.com/a/1650535
% This is the Proximal Operator for the case paramDelta = 1.
hProxHuberLoss1 = @(vX, paramLambda) vX - ((paramLambda * vX) ./ (max(abs(vX), paramLambda + 1)));
% Supporting any Huber Loss Function by Scaling of the Prox
% HuberLoss(vY, paramDelta) = paramDelta * paramDelta * HuberLoss(vY / paramDelta, 1)
hProxHuberLoss = @(vX, paramDelta, paramLambda) paramDelta * hProxHuberLoss1(vX / paramDelta, paramLambda);


%% Solution by CVX

solverString = 'CVX';

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numElements, 1);
    minimize( 0.5 * sum_square(vX - vY) + (0.5 * paramLambda * sum(huber(vX, paramDelta))) );
cvx_end

toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Analytic

solverString = 'Analytic';

tic();

% From https://math.stackexchange.com/a/1650535/33
vX = hProxHuberLoss(vY, paramDelta, paramLambda);

toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
% disp(['The ', solverString, ' Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Analytic

solverString = 'Analytic';

tic();

% By Boyd's Book
vX = ProxHuberLossBoyd(vY, paramDelta, paramLambda);

toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
% disp(['The ', solverString, ' Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

