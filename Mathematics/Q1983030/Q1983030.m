% Mathematics Q1983030
% https://math.stackexchange.com/questions/1983030
% Tikhonov Regularized Least Squares with Unit Simplex Constraint
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     22/08/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numRows = 4;
numCols = 3; %<! Number of Vectors - i (K in the question)

simplexRadius = 1;

paramLambda = 0.5;

numIterations = 5000;
stepSizeBase = 0.05;
stopThr = 1e-5;


%% Generate Data

mA = randn([numRows, numCols]);
vB = randn([numRows, 1]);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numCols)
    minimize( 0.5 * sum_square( mA * vX - vB ) + paramLambda * sum_square(vX) );
    subject to
        vX >= 0;
        sum(vX) == simplexRadius;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Projected Gradient Descent

vX = pinv(mA) * vB;

mAA = mA.' * mA;
mAb = mA.' * vB;

for ii = 1:numIterations
    stepSize = stepSizeBase / sqrt(ii);
    vG = (mAA * vX) - mAb + (2 * paramLambda * vX);
    vX = vX - (stepSize * vG);
    vX = ProjectSimplex(vX, simplexRadius, stopThr);
end

objVal = (0.5 * sum((mA * vX - vB) .^ 2)) + (paramLambda * sum(vX .^ 2));

disp([' ']);
disp(['Projected Gradient Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(objVal)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);




%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);
