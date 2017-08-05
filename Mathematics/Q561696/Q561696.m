% Mathematics Q561696
% https://math.stackexchange.com/questions/561696
% Solving Non Negative Least Squares by Analogy with Least Squares (MATLAB)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     05/08/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numRows = 4;
numCols = 3; %<! Number of Vectors - i (K in the question)

numIterations   = 10000;
stepSize        = 0.075;


%% Generate Data

mA = randn([numRows, numCols]);
vB = randn([numRows, 1]);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numCols)
    minimize( square_pos(  norm(mA * vX - vB, 2) ) );
    subject to
        vX >= 0;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Projected Gradient Descent

vX = zeros([numCols, 1]);

for ii = 1:numIterations
    vX = vX - ((stepSize / sqrt(ii)) * mA.' * (mA * vX - vB));
    vX = max(vX, 0);
end

objVal = sum((mA * vX - vB) .^ 2);

disp([' ']);
disp(['Projected Gradient Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(objVal)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

