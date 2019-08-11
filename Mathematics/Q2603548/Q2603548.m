% Mathematics Q2603548
% https://math.stackexchange.com/questions/2603548
% Solving Least Absolute Deviation (LAD) Line Fitting / Regression
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     11/08/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 0; % 2077;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Parameters

numRows = 7;
numCols = 3;

normP                       = 1;
numIterationsIrls           = 50;
epsVal                      = 1e-6;
numIterationsSubGradient    = 100000;
stepSize                    = 0.5e-5;

sLinProgSolverOpt = optimoptions('linprog', 'Display', 'off');


%% Load / Generate Data

mA = 5 * randn(numRows, numCols);
vB = 5 * randn(numRows, 1);

vXInit = pinv(mA) * vB;

hObjFun = @(vX) norm(mA * vX - vB, 1);


%% CVX Solution (Reference)

% Solution by CVX

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numCols, 1);
    minimize( norm(mA * vX - vB, 1) );
cvx_end

toc();

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp([' ']);

vXRef = vX;


%% Iteratively Reweighted Least Squares (IRLS) Solution

vX = vXInit;
vX(:) = 0;
mW = diag(1 ./ abs(mA * vX - vB));

for ii = 1:numIterationsIrls
    vX = (mA.' * mW * mA) \ (mA.' * mW * vB);
    mW = diag((abs(mA * vX - vB) + epsVal) .^ (normP - 2));
end

% vX = IRLS0(mA, vB, 1, numIterationsIrls);

disp([' ']);
disp('Iteratively Reweighted Least Squares (IRLS) Solution');
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Maximum Absolute Deviation - ', num2str(max(abs(vX - vXRef)))]);
disp([' ']);


%% Sub Gradient Method Solution

vX = vXInit;

for ii = 1:numIterationsSubGradient
    vG = mA.' * sign(mA * vX - vB);
    vX = vX - (stepSize * vG);
end

disp([' ']);
disp('Sub Gradient Method Solution');
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Maximum Absolute Deviation - ', num2str(max(abs(vX - vXRef)))]);
disp([' ']);


%% Linear Programming Conversion Solution

vTT = [ones(numRows, 1); zeros(numCols, 1)];
mAA = [-eye(numRows), -mA; -eye(numRows), mA];
vBB = [-vB; vB];
vX = linprog(vTT, mAA, vBB, [], [], [], [], sLinProgSolverOpt);

vX = vX((numRows + 1):(numRows + numCols));

disp([' ']);
disp('Linear Programming Conversion Solution');
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Maximum Absolute Deviation - ', num2str(max(abs(vX - vXRef)))]);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

