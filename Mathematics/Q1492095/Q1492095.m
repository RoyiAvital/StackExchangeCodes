% Mathematics Q1492095
% https://math.stackexchange.com/questions/1492095
% Orthogonal Projection on Intersection of Convex Sets
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     19/03/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

outOfSetThr     = 1e-6;
outOfSetCost    = 1e9;

mA = [-1, 1; 1, 0; 0, -1];
vB = [0; 2; 0];

ballRadius = 1;

numIterations   = 1000;
stopThr         = 1e-6;


%% Load / Generate Data

numRows = size(mA, 1);
numCols = size(mA, 2);
numSets = numRows + 1;

vY = 10 * randn(numCols, 1);

cProjFun = cell(numSets, 1);

for ii = 1:numRows
    cProjFun{ii} = @(vY) ProjectOntoHalfSpace(vY, mA(ii, :).', vB(ii));
end

cProjFun{numSets} = @(vY) min((ballRadius / norm(vY, 2)), 1) * vY;

hObjFun = @(vX) (0.5 * sum((vX - vY) .^ 2)) + (outOfSetCost * any(((mA * vX) - vB) > outOfSetThr)) + (outOfSetCost * (abs((vX.' * vX) - ballRadius) > outOfSetThr));


%% Solution by CVX

solverString = 'CVX';

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numCols, 1);
    minimize( 0.5 * sum_square(vX - vY) );
    subject to
        mA * vX <= vB;
        norm(vX) <= sqrt(ballRadius);
cvx_end

toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Alternating Projections

solverString = 'Alternating Projections';

tic();

vX = AlternatingProjectionOntoConvexSets(cProjFun, vY, numIterations, stopThr);

toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Linear Programming Solver (LP Solver)

solverString = 'LP';

tic();
vX = OrthogonalProjectionOntoConvexSets(cProjFun, vY, numIterations, stopThr);
toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by the Median (Analytic Solution)

solverString = 'Median (Analytic Solution)';

tic();
vX = median(mY, 2);
toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%%

numSamples = 1000;

vX1 = linspace(-2, 2, numSamples);
vX2 = linspace(-2, 2, numSamples);

mZ = zeros(numSamples);

for jj = 1:numSamples
    for ii = 1:numSamples
        mZ(ii, jj) = all(mA * [vX1(jj); vX2(ii)] <= vB);
    end
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

