% Test Projection onto L1 Ball
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
stopThr     = 1e-6;


%% Generating Data

vY = 10 * rand([numRows, 1]) - 5;

%% Solution by CVX

cvx_begin('quiet')
    % cvx_precision('best');
    variable vXCvx(numRows)
    minimize( norm(vXCvx - vY) )
    subject to
        norm(vXCvx ,1) <= ballRadius;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vXCvx.'), ' ]']);
disp([' ']);


%% Solution by Dual Function and Newton Iteration

% vX = ProjectL1Ball(vY, ballRadius, stopThr);
vX = ProjectL1BallExact(vY, ballRadius);

disp([' ']);
disp(['Dual Function Solution Summary']);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Display Results

disp([' ']);
disp(['CVX Solution L1 Norm - ', num2str(norm(vXCvx, 1))]);
disp(['Dual Function Solution L1 Norm - ', num2str(norm(vX, 1))]);
disp(['Solutions Difference L1 Norm - ', num2str(norm(vXCvx - vX, 1))]);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

