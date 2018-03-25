% Mathematics Q2706108
% https://math.stackexchange.com/questions/2706108
% Lasso ADMM with Positive Constraint
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     25/03/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

paramLambda = 0.5;

numRows = 12;
numCols = 8;

numIterations   = 250;
stepSize        = 0.0075;


%% Generate Data

mA = randn([numRows, numCols]);
vB = 10 * randn([numRows, 1]);

hObjFun = @(vX) (0.5 * sum(((mA * vX) - vB) .^ 2)) + (paramLambda * norm(vX, 1));
hL2SubGrad = @(vX) vX ./ max(norm(vX, 2), 1e-9);

hSoftThresholdL1 = @(vX, paramLambda) sign(vX) .* max(abs(vX) - paramLambda, 0);
hProjectRPlus = @(vX, paramLambda) max(vX, 0);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numCols)
    minimize( (0.5 * square_pos(norm((mA * vX) - vB, 2))) + (paramLambda * norm(vX, 1)) );
    subject to
        vX >= 0;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Sub Gradient Method

vObjValSgm = zeros([numIterations, 1]);

vX = zeros([numCols, 1]);
vObjValSgm(1) = hObjFun(vX);

mAA = mA.' * mA;
vAb = mA.' * vB;

for ii = 2:numIterations
    vG = (mAA * vX) - vAb + (paramLambda * sign(vX));
    vX = vX - (stepSize * vG);
    vX = max(vX, 0); %<! Project onto R+
    
    vObjValSgm(ii) = hObjFun(vX); 
end

disp([' ']);
disp(['Sub Gradient Method Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjValSgm(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution Proximal Projected Descent

vObjValPgm = zeros([numIterations, 1]);

[vX, mX] = SolveLsL1Prox(mA, vB, paramLambda, numIterations);

for ii = 1:numIterations    
    vObjValPgm(ii) = hObjFun(mX(:, ii));
end

disp([' ']);
disp(['Proximal Gradient Method Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjValPgm(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by ADMM

vObjValAdmm = zeros([numIterations, 1]);

[vX, mX] = SolveLsL1Admm(mA, vB, paramLambda, numIterations);

for ii = 1:numIterations
    vObjValAdmm(ii) = hObjFun(mX(:, ii));
end

disp([' ']);
disp(['ADMM Method Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjValAdmm(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);



%% Display Reesults

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot([1:numIterations], [cvx_optval * ones([numIterations, 1]), vObjValSgm, vObjValPgm, vObjValAdmm]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2:end), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', ['Least Squares with {L}_{1} Regularization (LASSO) with Positivity Constraint'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Index', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['CVX'], ['Sub Gradient Method'], ['Proximal Gradient Method'], ['ADMM']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

