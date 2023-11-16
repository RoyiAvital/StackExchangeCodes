% StackExchange Mathematics Q4804920
% https://math.stackexchange.com/questions/4804920
% Optimize Summation of L2 Norm and Infinity Norm.
% References:
%   1.  
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     12/11/2023  Royi Avital     RoyiAvital@yahoo.com
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
paramLambda     = 0.7;

% Solver Parameters
paramRho = 3;
numIterations = 100;



%% Load / Generate Data

vA = rand(numElements, 1); %<! The diagonal of mA (Positive)
mA = diag(vA);
vY = randn(numElements, 1);

% Should be paramLambda * max(mA * vX - vY) where mA is diagonal
hObjFun = @(vX) 0.5 * sum(vX .^ 2) + paramLambda * max(vA .* vX - vY);

solverIdx       = 0;
cMethodString   = {};

mObjFunValMse   = zeros([numIterations, 1]);
mSolMse         = zeros([numIterations, 1]);


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
    minimize( 0.5 * square_pos(norm(vX)) + paramLambda * norm(vA .* vX - vY, inf) )
cvx_end

runTime = toc(hRunTime);

DisplayRunSummary(solverString, hObjFun, vX, runTime, cvx_status);

sCvxSol.vXCvx     = vX;
sCvxSol.cvxOptVal = hObjFun(vX);



%% Solution by ADMM

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Solution by ADMM'];

hRunTime = tic();

[vX, mX] = SolveL2LInfAdmm(zeros(numElements, 1), mA, vY, paramLambda, paramRho, numIterations);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFun, vX, runTime);
[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFun, sCvxSol, solverIdx);


%% Display Results

figureIdx = figureIdx + 1;

hFigure = DisplayComparisonSummary(numIterations, mObjFunValMse, mSolMse, cLegendString, figPosLarge, lineWidthNormal, fontSizeTitle, fontSizeAxis);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end



%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

