% Test Projection onto L Inf Ball
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
        norm(vXCvx ,inf) <= ballRadius;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vXCvx.'), ' ]']);
disp([' ']);


%% Solution by Dual Function and Newton Iteration

vX = ProjectLInfBall(vY, ballRadius);

disp([' ']);
disp(['Closed Form Solution Summary']);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Display Results

disp([' ']);
disp(['CVX Solution L Inf Norm - ', num2str(norm(vXCvx, 'inf'))]);
disp(['Closed Form Solution L Inf Norm - ', num2str(norm(vX, 'inf'))]);
disp(['Solutions Difference L2 Norn - ', num2str(norm(vXCvx - vX, 2))]);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

