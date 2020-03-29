% Mathematics Q3599020
% https://math.stackexchange.com/questions/3599020
% Projection onto a Polyhedral Cone as Minimization of Different Norms
% References:
%   1.  See Isao Yamada Work:
%   https://scholar.google.com/citations?hl=en&user=InhJcBIAAAAJ,
%   https://ieeexplore.ieee.org/author/37085574318, https://ieeexplore.ieee.org/author/37282458700
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     30/03/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 2101; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;

COMP_METHOD_A = 1; %<! Faster Method
COMP_METHOD_B = 2;


%% Parameters

outOfSetThr     = 1e-5;
outOfSetCost    = 1e9;

numRows = 50;
numCols = 20;

numIterations   = 5e5;
stopThr         = outOfSetThr * outOfSetThr;


%% Load / Generate Data

mA = 2 * randn(numRows, numCols);
vB = zeros(numRows, 1);
vY = 2 * randn(numCols, 1);

% numSets = numRows + 1;
numSets = numRows;

cProjFun = cell(numSets, 1);

for ii = 1:numRows
    cProjFun{ii} = @(vY) ProjectOntoHalfSpace(vY, mA(ii, :).', vB(ii));
end

% cProjFun{numSets} = @(vY) min((sqrt(ballRadius) / norm(vY, 2)), 1) * vY;

hObjFun = @(vX) (0.5 * sum((vX - vY) .^ 2)) + (outOfSetCost * any(((mA * vX) - vB) > outOfSetThr)) + (outOfSetCost * ((vX.' * vX) - ballRadius > outOfSetThr));
hObjFun = @(vX) (0.5 * sum((vX - vY) .^ 2));


%% Solution by CVX

solverString = 'CVX';

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numCols, 1);
    minimize( 0.5 * sum_square(vX - vY) );
    subject to
        mA * vX <= vB;
        % norm(vX) <= sqrt(ballRadius);
cvx_end

toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

vXRef = vX;


%% Solution by Hybrid Projection Algorithm - Method A

solverString = 'Hybrid Projection Algorithm A';

tic();
vX = HybridOrthogonalProjectionOntoConvexSets(cProjFun, vY, numIterations, stopThr, COMP_METHOD_A);
toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Consensus ADMM

solverString = 'Consensus ADMM';

paramRho = 1;
cProxFun = cell(numSets + 1, 1);

cProxFun{1} = @(vV, paramRho) (vV + vY) / (1 + paramRho);

for ii = 2:numSets + 1
    cProxFun{ii} = @(vV, paramRho) cProjFun{ii - 1}(vV);
end

tic();
vX = ConsensusAdmm(cProxFun, numCols, paramRho, numIterations, stopThr);
toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

