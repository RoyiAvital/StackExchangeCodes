% Mathematics Q2301266
% https://math.stackexchange.com/questions/2301266
% Proximal Operator for $g\left(x\right)=\mu{\left\|x\right\|}_1 + I_{\left\|x\right\|_2 \leq 1} \left(x\right)$ ($L_1$ Norm and Unit Ball Constraint)
% Proximal Operator for Infinity Norm and Box Constraints.
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

paramAlpha = rand(1);


%% Load / Generate Data

vY      = 2 * randn(numElements, 1);
hObjFun = @(vX) 0.5 * sum((vX - vY) .^ 2) + (paramAlpha * max(abs(vX))) + (outOfSetCost * (any(vX > (1 + outOfSetThr)) || any(vX < (0 - outOfSetThr))));


%% Solution by CVX

solverString = 'CVX';

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numElements, 1);
    minimize( 0.5 * sum_square(vX - vY) + (paramAlpha * norm(vX, inf)) );
    subject to
        0 <= vX <= 1;
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

vX = ProxBoxIndicatorLInfNormReg(vY, paramAlpha);

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

