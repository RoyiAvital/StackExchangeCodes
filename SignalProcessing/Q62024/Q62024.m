% StackExchange Signal Processing Q62024
% https://dsp.stackexchange.com/questions/62024
% Proximal Gradient Method (PGM) for a Function Model with More than 2 Functions (Sum of Functions)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     25/11/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numElements = 40;
paramLambda1 = 0.1; %<! L1 Norm
paramLambda2 = 0.05; %<! TV Norm


%% Generate Data

vY = 10 * randn(numElements, 1);

% Generate the Diff Operator (1D Gradient) by Finite Differences
mD = spdiags([-ones(numElements, 1), ones(numElements, 1)], [0, 1], numElements - 1, numElements);

% Objective Function
hObjFun = @(vX) (0.5 * sum( (vX - vY) .^ 2)) + (paramLambda1 * sum(abs(vX))) + (paramLambda2 * sum(abs(mD * vX)));


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numElements);
    minimize( 0.5 * pow_pos(norm(vX - vY, 2), 2) + (paramLambda1 * norm(vX, 1)) + (paramLambda1 * norm(mD * vX, 1)));
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp([' ']);


%% Solution by Prox Function (Closed Form Solution)
%{
Solving
%}

vX = ProjectWeightedL2Ball(vY, mW, vC, paramLambda);

disp([' ']);
disp(['Projected Gradient Descent Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp([' ']);


%% Display Results

figureIdx = figureIdx + 1;

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

