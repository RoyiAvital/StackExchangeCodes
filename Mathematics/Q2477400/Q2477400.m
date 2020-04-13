% Mathematics Q3599003
% https://math.stackexchange.com/questions/2477400
% L1 Projection onto the Probability Simplex
% References:
%   1.  
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     30/03/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 0; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

outOfSetThr     = 1e-5;
outOfSetCost    = 1e9;

numElements = 10;


%% Load / Generate Data

vY = 0.005 * randn(numElements, 1);

hObjFun = @(vX) (sum(abs(vX - vY))) + (outOfSetCost * (abs(sum(vX) - 1) > outOfSetThr)) + (outOfSetCost * (any(vX < -outOfSetThr)));


%% Solution by CVX

solverString = 'CVX';

hRunTime = tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numElements, 1);
    minimize( norm(vX - vY, 1) );
    subject to
        sum(vX) == 1;
        vX >= 0;
cvx_end

runTime = toc(hRunTime);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);

vXRef = vX;


%% Solution by Linear Programming

solverString = 'Linear Programming';

hRunTime = tic();

vX = ProjectProbabilitySimplexL1(vY);

runTime = toc(hRunTime);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Solution by Linear Programming

solverString = 'Analytic';

hRunTime = tic();

vX = ProjectProbabilitySimplexL1Analytic(vY);

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

