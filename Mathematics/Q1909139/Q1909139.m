% StackExchange Mathematics Q1909139
% https://math.stackexchange.com/questions/1909139
% Projection of a Symmetric Matrix onto the Probability Simplex
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     17/06/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numRows = 4;


%% Generate Data

mY = randn(numRows, numRows);
% mY = mY + mY.';

vI = zeros(numRows * numRows, 1);
vI(1:(numRows + 1):(numRows ^ 2)) = 1;

hObjFun = @(mX) 0.5 * sum((mX(:) - mY(:)) .^ 2);


%% Solution by CVX

solverString = 'CVX';

% cvx_solver('SDPT3'); %<! Default, Keep numRows low
% cvx_solver('SeDuMi');
% cvx_solver('Mosek'); %<! Can handle numRows > 500, Very Good!
% cvx_solver('Gurobi');

hRunTime = tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable mX(numRows, numRows) symmetric;
    minimize( (0.5 * sum_square(mX(:) - mY(:))) );
    subject to
        % mX <In> symmetric(numRows); %<! Doesn't work for some reason
        trace(mX) == 1;
        mX >= 0;
cvx_end

runTime = toc(hRunTime);

DisplayRunSummary(solverString, hObjFun, mX, runTime, cvx_status);

sCvxSol.vXCvx     = mX(:);
sCvxSol.cvxOptVal = hObjFun(mX);


%% Solution by Orthogonal Projection onto the Intersection of Convex Sets

hP1 = @(vX) reshape((reshape(vX, numRows, numRows) + reshape(vX, numRows, numRows).') / 2, 1, numRows * numRows);
hP2 = @(vX) max(vX, 0); %<! Projection mX >= 0
hP3 = @(vX) (vX .* ~vI) + ((vX - ((sum(vX .* vI) - 1) / numRows)) .* vI); %<! Projection trace(mX) == 1

hRunTime = tic();
vX = OrthogonalProjectionOntoConvexSets({hP1; hP2; hP3}, mY(:), 100, 1e-6);
runTime = toc(hRunTime);

mX1 = reshape(vX, numRows, numRows);

DisplayRunSummary(solverString, hObjFun, reshape(vX, numRows, numRows), runTime);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

