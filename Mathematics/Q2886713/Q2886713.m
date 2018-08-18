% Mathematics Q2886713
% https://math.stackexchange.com/questions/2886713
% Prox Operator of L1 Norm with Linear Equality Constraint (Sum of Elements)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     18/08/2018
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numElements = 10;


%% Generate Data

vY = 10 * randn(numElements, 1);

paramGamma  = 0;
paramB      = 1;


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numElements);
    minimize( 0.5 * pow_pos(norm(vX - vY, 2), 2) + (paramGamma * norm(vX ,1)) );
    subject to
        sum(vX) == paramB;
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

hObjFun = @(vX) (0.5 * sum((vX - vY) .^ 2)) + (paramGamma * norm(vX ,1));

vXRef = vX;

vX = ProxL1NormSum(vY, paramGamma, paramB);

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

