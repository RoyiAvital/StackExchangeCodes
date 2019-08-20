% StackExchange Signal Processing Q60197
% https://dsp.stackexchange.com/questions/60197
% Implementation of Block Orthogonal Matching Pursuit (BOMP) Algorithm
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     19/08/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numRows         = 8;
% Pay attention: 'numBlocks * numElemntsBlock' should be larger than
% 'numRows'.
numBlocks       = 5;
numElemntsBlock = 3;

paramK = 3;
tolVal = 1e-6;


%% Generate Data

numCols = numBlocks * numElemntsBlock;

mA = randn(numRows, numCols);
vB = randn(numRows, 1);


%% Analysis

% Solving a Problem
vX = SolveLsL0Bomp(mA, vB, numBlocks, paramK, tolVal);

normErr = norm(mA * vX - vB);

disp([' ']);
disp(['Norm of the Error for A x - b - ', num2str(normErr)]);
disp([' ']);

% Comparing to OMP with the number of blocks matches the number of columns
mseErr = mean((SolveLsL0Bomp(mA, vB, numCols, paramK, tolVal) - SolveLsL0Omp(mA, vB, paramK, tolVal)) .^ 2);

disp([' ']);
disp(['MSE of the solution comparison between vanilla OMP to BOMP - ', num2str(mseErr)]);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

