% Mathematics Q2595199
% https://math.stackexchange.com/questions/2595199
% Proximal Mapping of Least Squares with L1 and L2 Mixed Norm Regularization (Elastic Net)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     13/03/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

paramLam1 = 1;
paramLam2 = 2;

numElements = 8;

numIterations   = 1000;
stepSize        = 0.0075;


%% Generate Data

vB = 10 * randn([numElements, 1]);

hObjFun = @(vX) (0.5 * sum((vX - vB) .^ 2)) + (paramLam1 * norm(vX, 1)) + (paramLam2 * norm(vX, 2));
hL2SubGrad = @(vX) vX ./ max(norm(vX, 2), 1e-9);

hSoftThresholdL1 = @(vX, paramLambda) sign(vX) .* max(abs(vX) - paramLambda, 0);
hSoftThresholdL2 = @(vX, paramLambda) vX .* (1 - (paramLambda / (max(norm(vX, 2), paramLambda))));


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numElements)
    minimize( (0.5 * square_pos(norm(vX - vB, 2))) + (paramLam1 * norm(vX, 1)) + (paramLam2 * norm(vX, 2)) );
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Analytic Solution

vX = hSoftThresholdL2(hSoftThresholdL1(vB, paramLam1), paramLam2);
analyticObjVal = hObjFun(vX);

disp([' ']);
disp(['Analytic Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(analyticObjVal)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Sub Gradient Method

vObjValSgm = zeros([numIterations, 1]);

vX = zeros([numElements, 1]);
vObjValSgm(1) = hObjFun(vX);

for ii = 2:numIterations
    vG = (vX - vB) + (paramLam1 * sign(vX)) + (paramLam2 * hL2SubGrad(vX));
    vX = vX - (stepSize * vG);
    
    vObjValSgm(ii) = hObjFun(vX); 
end

disp([' ']);
disp(['Sub Gradient Method Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjValSgm(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by https://doi.org/10.1007/s10957-012-0245-9 (A Primal Dual Splitting Method for Convex Optimization Involving Lipschitzian, Proximable and Linear Composite Terms)
% Implementing Algorithm 3.2
% The Matrix L is identity
% The Gradient of F is vX - vB

vObjValSplit = zeros([numIterations, 1]);

paramSigma  = 0.05;
paramTau    = 0.05;
paramPhi    = 0.5;

hSoftThresholdL1 = @(vX, paramLambda) sign(vX) .* max(abs(vX) - paramLambda, 0);
hSoftThresholdL2 = @(vX, paramLambda) vX .* (1 - (paramLambda / (max(norm(vX, 2), paramLambda))));

hProxG = @(vX, paramTau) hSoftThresholdL1(vX, paramTau * paramLam1); %<! Soft Thresholding L1
hProxH = @(vX, paramSigma) hSoftThresholdL2(vX, paramSigma * paramLam2); %<! Soft Thresholding L2

hProxHConj = @(vX, paramSigma) vX - (paramSigma * hProxH(vX / paramSigma, (1 / paramSigma))); %<! Moreau’s Identity

vX = zeros([numElements, 1]);
vY = zeros([numElements, 1]);
vXX = vX;
vYY = vY;

vObjValSplit(1) = hObjFun(vX);

for ii = 2:numIterations
    vYY = hProxHConj(vY + (paramSigma * vX), paramSigma * paramLam2);
    vXX = hProxG(vX - (paramTau * (vX - vB + (2 * vYY) - vY)), paramTau);
    
    vX = (paramPhi * vXX) + (1 - paramPhi) * vX;
    vY = (paramPhi * vYY) + (1 - paramPhi) * vY;
    
    vObjValSplit(ii) = hObjFun(vX);
end

disp([' ']);
disp(['Split Method Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjValSplit(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by ADMM (3 Blocks)
% Using Scaled Form (See Distributed Optimization and Statistical Learning
% via the Alternating Direction Method of Multipliers Pg. 15).

vObjValAdmm = zeros([numIterations, 1]);

paramRho = 0.25;

mA = [eye(numElements); eye(numElements)];
mB = [eye(numElements); zeros(numElements)];
mC = [zeros(numElements); eye(numElements)];

mI = eye(numElements);
mII = (1 / paramRho) * mI;
vBB = (1 / paramRho) * vB;

mAA = mA.' * mA;
mAB = mA.' * mB;
mAC = mA.' * mC;

mAAInv = inv(mAA + mII);

vX = zeros([numElements, 1]);
vY = zeros([numElements, 1]);
vZ = zeros([numElements, 1]);
vU = zeros([2 * numElements, 1]);

vObjValAdmm(1) = hObjFun(vX);

for ii = 2:numIterations
    
    vX = mAAInv * (vBB + (mAB * vY) + (mAC * vZ) - (mA.' * vU));
    % vX = (mAA + mII) \ (vBB + (mAB * vY) + (mAC * vZ) - (mA.' * vU));
    vY = SolveLsL1Prox(mB, ((mA * vX) + vU - (mC * vZ)), (paramLam1 / paramRho), numIterations);
    vZ = SolveLsL2Prox(mC, ((mA * vX) + vU - (mB * vY)), (paramLam2 / paramRho), numIterations);
    
    vU = vU + (mA * vX) - (mB * vY) - (mC * vZ);
    
    vObjValAdmm(ii) = hObjFun(vX);
end

disp([' ']);
disp(['ADMM 3 Blocks Method Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjValAdmm(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);



%% Display Reesults

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot([1:numIterations], [cvx_optval * ones([numIterations, 1]), analyticObjVal * ones([numIterations, 1]), vObjValSgm, vObjValSplit, vObjValAdmm]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2:end), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', ['Least Squares with Mixed Norm Regularization'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Index', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['CVX'], ['Analytic Solution'], ['Sub Gradient Method'], ['Split Method'], ['ADMM - 3 Blocks']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

