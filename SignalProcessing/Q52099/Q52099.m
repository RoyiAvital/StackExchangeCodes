% StackExchange Signal Processing Q52099
% https://dsp.stackexchange.com/questions/52099
% How to Formulate a Constraint Which Ensures All Variables Have the Same Sign
% References:
%   1.  
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     19/04/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 0; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

numRows = 8;
numCols = 6; %<! Half Positive, Half Negative

boundRadius = 1e3;
lowerBound  = -boundRadius;
upperBound  = boundRadius;


%% Load / Generate Data

mA = randn(numRows, numCols);

vXRef                                   = randn(numCols, 1);
vXRef(1:floor(numCols / 2))             = abs(vXRef(1:floor(numCols / 2)));
vXRef(ceil((numCols + 1) / 2):numCols)  = -abs(vXRef(ceil((numCols + 1) / 2):numCols));

vB = mA * vXRef;

hObjFun = @(vX) sum(((mA * vX) - vB) .^ 2);


%% Solution by CVX

solverString = 'CVX';

cvx_solver('SDPT3'); %<! Default
% cvx_solver('SeDuMi');
% cvx_solver('Mosek');
% cvx_solver('Gurobi');

hRunTime = tic();

cvx_begin('quiet')
% cvx_begin()
    % cvx_precision('best');
    variable vX(numCols, 1);
    variable varY(1) binary;
    minimize( norm(mA * vX - vB, 2) );
    subject to
        vX >= (1 - varY) * lowerBound;
        vX <= varY * upperBound;
cvx_end

runTime = toc(hRunTime);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Solution by Non Negative Least Squares

solverString = 'Non Negative Least Squares';

hRunTime = tic();

vX = SolveLsSameSign(mA, vB);

runTime = toc(hRunTime);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

