% Test Projection onto L2 Ball
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     27/06/2017  Royi Avital
%   *   First release.


%% General Parameters

run('InitScript.m');

numRows     = 5;
ballRadius  = 3;


%% Generating Data

vY = 10 * rand([numRows, 1]) - 5;

%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vXCvx(numRows)
    minimize( norm(vXCvx - vY) )
    subject to
        norm(vXCvx ,2) <= ballRadius;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vXCvx.'), ' ]']);
disp([' ']);


%% Solution by Dual Function and Newton Iteration

vX = ProjectL2Ball(vY, ballRadius);

disp([' ']);
disp(['Closed Form Solution Summary']);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Display Results

disp([' ']);
disp(['CVX Solution L2 Norm - ', num2str(norm(vXCvx, 2))]);
disp(['Dual Function Solution L2 Norm - ', num2str(norm(vX, 2))]);
disp(['Solutions Difference L2 Norn - ', num2str(norm(vXCvx - vX, 2))]);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

