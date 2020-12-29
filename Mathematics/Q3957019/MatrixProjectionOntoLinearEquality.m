% StackExchange Mathematics Q3957019
% https://math.stackexchange.com/questions/3957019
% Solving 2 kind of projections:
% 1.    arg min_X 0.5 * || X - Y ||_F^2 s. t. X a = b.
% 1.    arg min_X 0.5 * || X - Y ||_F^2 s. t. X' a = b.
% References:
%   1.  
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     25/12/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 0; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

numRows = 12;
numCols = 10;


%% Load / Generate Data

mY = randn(numRows, numCols);
vA = randn(numCols, 1);
vB = randn(numRows, 1);

hObjFun = @(mX) 0.5 * (norm(mX - mY, 'fro') ^ 2);


%% Solution by CVX

solverString = 'CVX';

% cvx_solver('SDPT3'); %<! Default, Keep numRows low
% cvx_solver('SeDuMi');
% cvx_solver('Mosek'); %<! Can handle numRows > 500, Very Good!
% cvx_solver('Gurobi');

hRunTime = tic();

cvx_begin('quiet')
% cvx_begin()
    % cvx_precision('best');
    variable mX(numRows, numCols);
    minimize( 0.5 * square_pos(norm(mX - mY, 'fro')) );
    subject to
        mX * vA == vB;
cvx_end

runTime = toc(hRunTime);

% vX = mX(:);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(mX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mX(:).'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Solution by Analytic Solution

solverString = 'Solution by Analytic Solution';

hRunTime = tic();

mX = mY - ((((mY * vA) - vB) / (vA.' * vA)) * vA.');

runTime = toc(hRunTime);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(mX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mX(:).'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Solution by CVX

% Switching in order to match dimensions
vT = vA;
vA = vB;
vB = vT;

solverString = 'CVX';

% cvx_solver('SDPT3'); %<! Default, Keep numRows low
% cvx_solver('SeDuMi');
% cvx_solver('Mosek'); %<! Can handle numRows > 500, Very Good!
% cvx_solver('Gurobi');

hRunTime = tic();

cvx_begin('quiet')
% cvx_begin()
    % cvx_precision('best');
    variable mX(numRows, numCols);
    minimize( 0.5 * square_pos(norm(mX - mY, 'fro')) );
    subject to
        mX.' * vA == vB;
cvx_end

runTime = toc(hRunTime);

% vX = mX(:);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(mX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mX(:).'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Solution by Analytic Solution

solverString = 'Solution by Analytic Solution';

hRunTime = tic();

mX = mY - (vA * ( ((mY.' * vA) - vB) / (vA.' * vA) ).');

runTime = toc(hRunTime);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(mX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mX(:).'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

