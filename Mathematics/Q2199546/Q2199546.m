% Mathematics Q2199546
% https://math.stackexchange.com/questions/2199546
% Minimizing the Sum of Quadratic Form with Equality Constraint
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     29/07/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numRows = 4;
numCols = 3; %<! Number of Vectors - i (K in the question)

numIterations   = 1000;
stepSize        = 0.05;


%% Generate Data

vE          = ones([numRows, 1]);
tA          = randn([numRows, numRows, numCols]);
mC          = randn([numRows, numCols]);
paramLambda = 1;

for ii = 1:numCols
    tA(:, :, ii) = tA(:, :, ii).' * tA(:, :, ii);
end

hProjEquality = @(mX) mX - ((sum(mX, 2) - vE) / numCols);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable mXCvx(numRows, numCols)
    objVal = quad_form( mXCvx(:, 1), tA(:, :, 1) ) + (paramLambda * (mC(:, 1).' * mXCvx(:, 1)));
    for ii = 2:numCols
        objVal = objVal + quad_form( mXCvx(:, ii), tA(:, :, ii) ) + (paramLambda * (mC(:, ii).' * mXCvx(:, ii)));
    end
    minimize( objVal )
    subject to
        mXCvx * ones([numCols, 1]) == vE; %<! vE to be Eigen Vector for Eigen Value 1
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mXCvx(:).'), ' ]']);
disp([' ']);


%% Solution by Projected Gradient Descent

mX = zeros([numRows, numCols]);

for ii = 1:numIterations
    for jj = 1:numCols
        mX(:, jj) = mX(:, jj) - (stepSize * ((2 * tA(:, :, jj) * mX(:, jj)) + (paramLambda * mC(:, jj))));
    end
    mX = hProjEquality(mX);
end

objVal = 0;
for ii = 1:numCols
    objVal = objVal + (mX(:, ii).' * tA(:, :, ii) * mX(:, ii)) + (paramLambda * mC(:, ii).' * mX(:, ii));
end

disp([' ']);
disp(['Projected Gradient Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(objVal)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mX(:).'), ' ]']);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

