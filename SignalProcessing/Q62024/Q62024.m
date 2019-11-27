% StackExchange Signal Processing Q62024
% https://dsp.stackexchange.com/questions/62024
% Proximal Gradient Method (PGM) for a Function Model with More than 2 Functions (Sum of Functions)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     25/11/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numRows = 40;
numCols = 30;
paramLambda1 = 0.5; %<! L1 Norm
paramLambda2 = 0.15; %<! TV Norm

numIterations   = 15000;
stepSize        = 0.00015;

numIterationsAdmm = 50;

cMethodNames = {['CVX'], ['Sub Gradient Method'], ['Proximal Gradient Method']};
% cMethodNames = {['CVX'], ['Sub Gradient Method'], ['Proximal Gradient Method'], ['ADMM']};
methodIdx = 0;


%% Generate Data

mA = randn(numRows, numCols);
vY = 10 * randn(numRows, 1);

vAy = mA.' * vY;
mAA = mA.' * mA;

% Generate the Diff Operator (1D Gradient) by Finite Differences
mD = spdiags([-ones(numCols, 1), ones(numCols, 1)], [0, 1], numCols - 1, numCols);

% Objective Function
hObjFun = @(vX) (0.5 * sum( (mA * vX - vY) .^ 2)) + (paramLambda1 * sum(abs(vX))) + (paramLambda2 * sum(abs(mD * vX)));

% The PRox Function
hProxFunction = @(vY, paramLambda1, paramLambda2) SolveProxTvAdmm(SolveProxL1(vY, paramLambda1), mD, paramLambda2, numIterationsAdmm);

numMethods  = size(cMethodNames, 1);
mObjVal     = zeros(numIterations, numMethods);


%% Solution by CVX

methodIdx = methodIdx + 1;

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numCols);
    minimize( (0.5 * pow_pos(norm(mA * vX - vY, 2), 2)) + (paramLambda1 * norm(vX, 1)) + (paramLambda2 * norm(mD * vX, 1)));
cvx_end

disp([' ']);
disp([cMethodNames{methodIdx}, ' Solution Summary']);
disp(['The ', cMethodNames{methodIdx}, ' Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp([' ']);

mObjVal(:, methodIdx) = cvx_optval * ones([numIterations, 1]);


%% Solution by Sub Gradient Descent
%{
Solving $ \arg \min_x \frac{1}{2} {\left\| x - y \right\|}_{2}^{2} +
{\lambda}_{1} {\left\| x \right\|}_{1} + {\lambda}_{2} {\left\| D x \right\|}_{1} $
%}

methodIdx = methodIdx + 1;

vX = pinv(mA) * vY;
vG = vX;
mObjVal(1, methodIdx) = hObjFun(vX);

for ii = 2:numIterations
    vG(:) = (mAA * vX - vAy) + (paramLambda1 * sign(vX)) + (paramLambda2 * mD.' * sign(mD * vX));
    vX(:) = vX - (stepSize * vG);
    mObjVal(ii, methodIdx) = hObjFun(vX);
end

disp([' ']);
disp([cMethodNames{methodIdx}, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp([' ']);


%% Solution by Proximal Gradient Method (PGM)
%{
Solving
%}

methodIdx = methodIdx + 1;

stepSize = 1 / (2 * (norm(mA, 2) ^ 2));

vX = pinv(mA) * vY;
vG = vX;
mObjVal(1, methodIdx) = hObjFun(vX);

for ii = 2:numIterations
    vG(:) = mAA * vX - vAy;
    vX(:) = hProxFunction(vX - (stepSize * vG), paramLambda1 * stepSize, paramLambda2 * stepSize);
    % vX(:) = SolveProxL1(vX - (stepSize * vG), stepSize * paramLambda1);
    mObjVal(ii, methodIdx) = hObjFun(vX);
end

disp([' ']);
disp([cMethodNames{methodIdx}, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp([' ']);


%% Display Results

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot([1:numIterations], mObjVal);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2:end), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Least Squares with {L}_{1} (LASSO) and Total Varitaion Norm Regularization'], ['SubStream - ', num2str(subStreamNumber)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Index', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend(cMethodNames);
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

