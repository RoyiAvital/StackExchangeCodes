% StackExchange Mathematics Q3972913
% https://math.stackexchange.com/questions/3972913
% Orthogonal Projection onto a Variation of the Unit Simplex
% References:
%   1.  
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     05/01/2021  Royi Avital     RoyiAvital@yahoo.com
%   *   First release.


%% General Parameters

subStreamNumberDefault = 0; 42; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

% Problem parameters
numElements     = 10;
simplexRadius   = 2;
paramAlpha      = simplexRadius / numElements; %<! Minimum Feasible Value
paramAlpha      = simplexRadius; %<! Maximum effective Value
paramAlpha      = (simplexRadius / numElements) + (simplexRadius - (simplexRadius / numElements)) * rand(1, 1); %<! Random

% Solver parameters
stopThr = 1e-6;


%% Load / Generate Data

vY = randn(numElements, 1);

hObjFun = @(vX) 0.5 * sum((vX - vY) .^ 2);

solverIdx       = 0;
cMethodString   = {};


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
    variable vX(numElements, 1);
    minimize( 0.5 * square_pos(norm(vX - vY)) );
    subject to
        sum(vX) == simplexRadius;
        vX >= 0;
        vX <= paramAlpha;
cvx_end

runTime = toc(hRunTime);

DisplayRunSummary(solverString, hObjFun, vX, runTime, cvx_status);

sCvxSol.vXCvx     = vX;
sCvxSol.cvxOptVal = hObjFun(vX);


%% Solution by Dual Function

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Solution by Dual Function'];

hRunTime = tic();

vX = ProjectSimplexBox(vY, simplexRadius, paramAlpha, stopThr);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFun, vX, runTime);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

