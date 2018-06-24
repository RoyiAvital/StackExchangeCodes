% Mathematics Q2824418
% https://math.stackexchange.com/questions/2824418
% Projection Operator onto L1 Ball with Box Constraints
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.1.000     24/06/2018
%   *   Added Dual Function solver.
%   *   Added Objective Function value.
% - 1.0.000     22/06/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';
generateFigures     = ON;


%% Simulation Parameters

numRows     = 5;
ballRadius  = 2.9;


%% Generate Data

vY          = 10 * rand([numRows, 1]) - 5;
vLowerBound = -2 * rand([numRows, 1]);
vUpperBound = vLowerBound + (1 * rand([numRows, 1]));

hObjFun = @(vX) 0.5 * sum((vX - vY) .^ 2);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vXCvx(numRows)
    minimize( 0.5 * sum_square_pos(norm(vXCvx - vY)) )
    subject to
        norm(vXCvx ,1) <= ballRadius;
        vXCvx >= vLowerBound;
        vXCvx <= vUpperBound;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vXCvx.'), ' ]']);
disp([' ']);


%% Solution by KKT Conditions

vX = ProjectL1Ball(vY, ballRadius, vLowerBound, vUpperBound);

disp([' ']);
disp(['KKT Function Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Dual Function

vX = ProjectL1BallDual(vY, ballRadius, vLowerBound, vUpperBound);

disp([' ']);
disp(['Dual Function Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Display Results

disp([' ']);
disp(['CVX Solution L1 Norm - ', num2str(norm(vXCvx, 1))]);
disp(['KKT Function Solution L1 Norm - ', num2str(norm(vX, 1))]);
disp(['Solutions Difference L1 Norn - ', num2str(norm(vXCvx - vX, 1))]);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

