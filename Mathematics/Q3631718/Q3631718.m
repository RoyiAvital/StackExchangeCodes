% StackExchange Mathematics Q3631718
% https://math.stackexchange.com/questions/3631718
% Find a Symmetric Matrix N to Minimize ∥N−M∥2F with Constraint Nd=g
% References:
%   1.  
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     21/04/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 0; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 79;
figureCounterSpec   = '%04d';

generateFigures = ON;

EXTRACT_LOWER_TRIANGLE = 1;
EXTRACT_UPPER_TRIANGLE = 2;

INCLUDE_DIAGONAL = 1;
EXCLUDE_DIAGONAL = 2;


%% Parameters

numRows = 5; %<! Symmetric Matrix


%% Load / Generate Data

mY = randn(numRows, numRows);
vA = randn(numRows, 1);
vB = randn(numRows, 1);

hObjFun = @(vX) 0.5 * sum((vX - mY(:)) .^ 2); %<! Frobenius Norm


%% Solution by CVX

solverString = 'CVX';

cvx_solver('SDPT3'); %<! Default, Keep numRows low
% cvx_solver('SeDuMi');
% cvx_solver('Mosek'); %<! Can handle numRows > 500, Very Good!
% cvx_solver('Gurobi');

hRunTime = tic();

cvx_begin('quiet')
% cvx_begin()
    % cvx_precision('best');
    variable mX(numRows, numRows) symmetric;
    minimize( norm(mX - mY, 'fro') );
    subject to
        % mX == symmetric(numRows);
        mX * vA == vB;
cvx_end

runTime = toc(hRunTime);

vX = mX(:);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Solution by Vectrorizing the Problem
% See https://math.stackexchange.com/a/3634876.

solverString = 'Closed Form Solution - Vectrorizing the Problem';

hRunTime = tic();

mLU = GenerateSymmetricConstraintMatrix(numRows);
mD = kron(vA.', speye(numRows));

mC = [mLU; mD];

numElementsMu = size(mLU, 1);

% vX = [speye(numRows * numRows), mC.'; mC, spalloc(numElementsMu + numRows, numElementsMu + numRows, 0)] \ [mY(:); zeros(numElementsMu, 1); vB];
% vX = [speye(numRows * numRows), mC.'; mC, 1e-5 * speye(numElementsMu + numRows)] \ [mY(:); zeros(numElementsMu, 1); vB];

% See Iterative Solvers - https://www.mathworks.com/help/matlab/math/iterative-methods-for-linear-systems.html.
vX = minres([speye(numRows * numRows), mC.'; mC, spalloc(numElementsMu + numRows, numElementsMu + numRows, 0)], [mY(:); zeros(numElementsMu, 1); vB]);
% vX = symmlq([speye(numRows * numRows), mC.'; mC, spalloc(numElementsMu + numRows, numElementsMu + numRows, 0)], [mY(:); zeros(numElementsMu, 1); vB]);
% vX = lsqr([speye(numRows * numRows), mC.'; mC, spalloc(numElementsMu + numRows, numElementsMu + numRows, 0)], [mY(:); zeros(numElementsMu, 1); vB]);

vX = vX(1:(numRows * numRows));

runTime = toc(hRunTime);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Solution by Symmetric Sum
% See https://math.stackexchange.com/a/3635261.

solverString = 'Closed Form Solution - Symmetric Sum';

hRunTime = tic();

vV = (((vA.' * vA) * eye(numRows)) + (vA * vA.')) \ ((mY + mY.') * vA - (2 * vB));
mX = 0.5 * ((mY + mY.') - (vV * vA.') - (vA * vV.'));

runTime = toc(hRunTime);

vX = mX(:);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

