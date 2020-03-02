% Mathematics Q3566493
% https://math.stackexchange.com/questions/3566493
% The Minimizer of $ {L}_{1} $ Norm for a Set of Vectors
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     02/03/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

vecDim      = 5;
numSamples  = 20;


%% Load / Generate Data

mY = randn(vecDim, numSamples) + (5 * rand(vecDim, numSamples));
hObjVal = @(vX) sum(vecnorm(mY - vX, 1, 1));


%% Solution by CVX

solverString = 'CVX';

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(vecDim, 1);
    objVal = 0;
    for ii = 1:numSamples
        objVal = objVal + norm(mY(:, ii) - vX, 1);
    end
    % For 'norms()' see http://ask.cvxr.com/t/4351 and http://cvxr.com/cvx/doc/funcref.html
    % For loop see http://ask.cvxr.com/t/5088
    minimize( objVal );
cvx_end

toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Value Is Given By - ', num2str(hObjVal(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by CVX

solverString = 'CVX';

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(vecDim, 1);
    % For 'norms()' see http://ask.cvxr.com/t/4351 and http://cvxr.com/cvx/doc/funcref.html
    % For loop see http://ask.cvxr.com/t/5088
    minimize( sum(norms(mY - (vX * ones(numSamples, 1).'), 1, 1)) );
cvx_end

toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Value Is Given By - ', num2str(hObjVal(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Linear Programming Solver (LP Solver)

solverString = 'LP';

tic();
vX = SolveL1NormSetMinimization(mY);
toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjVal(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by the Median (Analytic Solution)

solverString = 'Median (Analytic Solution)';

tic();
vX = median(mY, 2);
toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjVal(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

