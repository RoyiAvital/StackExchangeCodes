% Mathematics Q1639716
% https://math.stackexchange.com/questions/2706108
% How Can L1 Norm Minimization with Linear Equality Constraints (Basis Pursuit / Sparse Representation) Be Formulated as Linear Programming?
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     05/04/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numRows = 10;
numCols = 3 * numRows;


%% Generate Data

mA = randn([numRows, numCols]);
vB = 10 * randn([numRows, 1]);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numCols)
    minimize( norm(vX, 1) );
    subject to
        mA * vX == vB;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Linear Progeamming - Method A

vX = SolveBasisPursuitLp001(mA, vB);

disp([' ']);
disp(['Linear Progeamming - Method A Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(norm(vX, 1))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Linear Progeamming - Method B

vX = SolveBasisPursuitLp002(mA, vB);

disp([' ']);
disp(['Linear Progeamming - Method B Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(norm(vX, 1))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

