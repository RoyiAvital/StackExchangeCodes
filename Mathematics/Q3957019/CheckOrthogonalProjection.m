% StackExchange Mathematics Q3957019
% https://math.stackexchange.com/questions/3957019
% Check the orthogonal projection
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

COMP_METHOD_A = 1; %<! Faster Method
COMP_METHOD_B = 2;


%% Parameters

userDataFileName = 'UserData.mat';

numIterations   = 700000;
stopThr         = 1e-9;


%% Load / Generate Data

load(userDataFileName);

numRows = size(mK2, 2);
numCols = size(mK1, 2);

mY = randn(numRows, numCols);

vOnesRows = ones(numRows, 1);
vOnesCols = ones(numCols, 1);

hObjFun = @(mX) 0.5 * sum((mX(:) - mY(:)) .^ 2);

% See MatrixProjectionOntoLinearEquality for the case vA = ones()
hProjEqualConst1    = @(mS) mS - (vOnesRows * ( (sum(mS, 1) - vU.') / (numRows) )); %<! Sum in the 1st dimension, Can be more efficient with rempat()
hProjEqualConst2    = @(mS) mS - (((sum(mS, 2) - vV) / (numCols)) * vOnesCols.'); %<! Sum in the 2nd dimension, Can be more efficient with rempat()
hProjInequalConst   = @(mS) max(mS, 0);

hProjEqualConst1Vec = @(vS) reshape(hProjEqualConst1(reshape(vS, numRows, numCols)), numRows * numCols, 1);
hProjEqualConst2Vec = @(vS) reshape(hProjEqualConst2(reshape(vS, numRows, numCols)), numRows * numCols, 1);
hProjInequalConstVec = @(vS) reshape(hProjInequalConst(reshape(vS, numRows, numCols)), numRows * numCols, 1);

cProjFun = {hProjEqualConst1Vec; hProjEqualConst2Vec; hProjInequalConstVec};

solverIdx       = 0;
cMethodString   = {};


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
    variable mX(numRows, numCols);
    minimize( 0.5 * square_pos(norm(mX - mY, 'fro')) )
    subject to
        sum(mX.', 2) == vU;
        sum(mX, 2) == vV;
        mX >= 0;
cvx_end

runTime = toc(hRunTime);

mXRef = mX;

DisplayRunSummary(solverString, hObjFun, mX, runTime, cvx_status);


%% Solution by Dykstra's Projection Algorithm

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Dykstra''s Projection Algorithm'];

hRunTime = tic();

% mX = reshape(OrthogonalProjectionOntoConvexSets(cProjFun, mY(:), numIterations, stopThr), numRows, numCols);
% mX = reshape(HybridOrthogonalProjectionOntoConvexSets(cProjFun, mY(:), numIterations, stopThr, COMP_METHOD_A), numRows, numCols);
mX = reshape(OrthogonalProjectionOntoConvexSetsAdmm(cProjFun, mY(:), numIterations, stopThr), numRows, numCols);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFun, mX, runTime);


drawnow();

max(abs(sum(mXRef.', 2) - vU))
max(abs(sum(mXRef, 2) - vV))
all(full(mXRef(:)) >= 0)
max(abs(sum(mX.', 2) - vU))
max(abs(sum(mX, 2) - vV))
all(mX(:) >= 0)


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

