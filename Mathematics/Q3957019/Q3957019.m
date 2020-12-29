% StackExchange Mathematics Q3957019
% https://math.stackexchange.com/questions/3957019
% Solve Matrix Least Squares with Frobenius Norm Regularization with Linear Equality and Linear Inequality Constraints
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

subStreamNumberDefault = 42; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

userDataFileName = 'UserData.mat';

% Solvers parameters
numIterations       = 7500;
stepSizeGd          = 5e-5;
stepSizeMomentum    = 0.8;
stepSizeAccel       = 0.0025;

% Projection Parameters
numIterationsProj   = 5000;
stopThrProj         = 1e-7;


%% Load / Generate Data

load(userDataFileName);
% Make sure sum(vV) == sum(vU). Otherwise the problem is infeasible.
vU(1) = vU(1) + (sum(vV) - sum(vU));

numRows = size(mK2, 2);
numCols = size(mK1, 2);
numElmn = numRows * numCols;

mKK1 = mK1.' * mK1;
mKK2 = mK2.' * mK2;
mK2MK1 = mK2.' * mM * mK1;

vOnesRows = ones(numRows, 1);
vOnesCols = ones(numCols, 1);

hObjFun = @(mS) 0.5 * (norm((mK2 * mS * mK1.') - mM, 'fro') .^ 2) + (0.5 * paramLambda * (mS(:).' * mS(:)));
% hG = @(mS) (mKK2 * mS * mKK1) - mK2MK1 + (paramLambda * mS);
hG = @(mS) mK2.' * ((mK2 * mS * mK1.') - mM) * mK1 + (paramLambda * mS);
% See MatrixProjectionOntoLinearEquality for the case vA = ones()
hProjEqualConst1    = @(mS) mS - (vOnesRows * ( (sum(mS, 1) - vU.') / (numRows) )); %<! Sum in the 1st dimension, Can be more efficient with rempat()
hProjEqualConst2    = @(mS) mS - (((sum(mS, 2) - vV) / (numCols)) * vOnesCols.'); %<! Sum in the 2nd dimension, Can be more efficient with rempat()
hProjInequalConst   = @(mS) max(mS, 0);

hGVec       = @(vS) reshape(hG(reshape(vS, numRows, numCols)), numElmn, 1);
hObjFunVec  = @(vS) hObjFun(reshape(vS, numRows, numCols));

hProjEqualConst1Vec = @(vS) reshape(hProjEqualConst1(reshape(vS, numRows, numCols)), numElmn, 1);
hProjEqualConst2Vec = @(vS) reshape(hProjEqualConst2(reshape(vS, numRows, numCols)), numElmn, 1);
hProjInequalConstVec = @(vS) reshape(hProjInequalConst(reshape(vS, numRows, numCols)), numElmn, 1);

cProjFun = {hProjEqualConst1Vec; hProjEqualConst2Vec; hProjInequalConstVec};

hP = @(mS) reshape(OrthogonalProjectionOntoConvexSets(cProjFun, mS(:), numIterationsProj, stopThrProj), numRows, numCols);

solverIdx       = 0;
cMethodString   = {};

mObjFunValMse   = zeros([numIterations, 1]);
mSolMse         = zeros([numIterations, 1]);


%% Solution by CVX

solverString = 'CVX';

% cvx_solver('SDPT3'); %<! Default, Keep numRows low
% cvx_solver('SeDuMi');
cvx_solver('Mosek'); %<! Can handle numRows > 500, Very Good!
% cvx_solver('Gurobi');

hRunTime = tic();

cvx_begin('quiet')
% cvx_begin()
    % cvx_precision('best');
    variable mS(numRows, numCols);
    minimize( 0.5 * square_pos(norm((mK2 * mS * mK1.') - mM, 'fro')) + (0.5 * paramLambda * square_pos(norm(mS, 'fro'))) );
    subject to
        sum(mS.', 2) == vU;
        sum(mS, 2) == vV;
        mS >= 0;
cvx_end

runTime = toc(hRunTime);

DisplayRunSummary(solverString, hObjFun, mS, runTime, cvx_status);

sCvxSol.vXCvx     = mS(:);
sCvxSol.cvxOptVal = hObjFun(mS);


%% Solution by Projected Gradient Descent

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Projected Gradient Method'];

hRunTime = tic();

[vX, mX] = ProjectedGd(zeros(numElmn, 1), hGVec, hP, numIterations, stepSizeGd);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFunVec, vX, runTime);

[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFunVec, sCvxSol, solverIdx);


%% Solution by Projected Gradient Descent with Momentum

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Projected Gradient Descent with Momentum'];

hRunTime = tic();

[vX, ~, mX] = ProjectedGdMomentum(zeros(numElmn, 1), zeros(numElmn, 1), hGVec, hP, numIterations, stepSizeGd, stepSizeMomentum);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFunVec, vX, runTime);

[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFunVec, sCvxSol, solverIdx);


%% Solution by Projected Gradient Descent with Nesterov Acceleration

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Projected Gradient Descent with Nesterov Acceleration'];

hRunTime = tic();

[vX, ~, mX] = ProjectedGdAccel(zeros(numElmn, 1), zeros(numElmn, 1), hGVec, hP, numIterations, stepSizeGd, stepSizeAccel);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFunVec, vX, runTime);

[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFunVec, sCvxSol, solverIdx);


%% Solution by Projected Gradient Descent with FISTA

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Projected Gradient Descent with FISTA Acceleration'];

hRunTime = tic();

[vX, ~, mX] = ProjectedGdFista(zeros(numElmn, 1), zeros(numElmn, 1), hGVec, hP, numIterations, stepSizeGd);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFunVec, vX, runTime);

[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFunVec, sCvxSol, solverIdx);


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

