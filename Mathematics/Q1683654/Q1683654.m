% Mathematics Q1683654
% https://math.stackexchange.com/questions/1683654
% Proximal / Prox Operator for the Logistic Function
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     06/03/2020
%   *   First release.


%% General Parameters

% subStreamNumber which fails: 2084, 2122
subStreamNumberDefault = 0; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numElements = 7;
paramLambda = 0.5;


%% Generate Data

vC = randn(numElements, 1);
vY = randn(numElements, 1);

hObjFun = @(vX) 0.5 * sum((vX - vY) .^ 2) + (paramLambda * log(1 + exp(-vC.' * vX)));

numIterations   = 100;
stopThr         = 1e-6;
setpSize        = 1e-4;
vX0             = zeros(numElements, 1);


%% Solution by CVX

solverString = 'CVX';

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numElements, 1);
    minimize( (0.5 * sum_square(vX - vY)) + (paramLambda * log(1 + exp(-vC.' * vX))) );
cvx_end

toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Fixed Point Iteration

solverString = 'Fixed Point Iteration';

tic()
vX = ProxLogisticLossFunction(vX0, vY, vC, paramLambda, numIterations, stopThr);
toc()

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Newton Method

solverString = 'Newton Method';

sSolverOptions = optimoptions('fminunc', 'Display', 'off');
tic()
vX = fminunc(hObjFun, vX0, sSolverOptions);
toc()

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Gradient Descent

solverString = 'Gradient Descent';

tic()
vX = ProxLogisticLossFunctionGd(vX0, vY, vC, paramLambda, 100 * numIterations, stopThr);
toc()

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

