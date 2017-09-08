% Mathematics Q2403596
% https://math.stackexchange.com/questions/2403596
% Least Square Linear Regression with P Norm Regularization Where 1?P?2
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     08/09/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;


%% Simulation Parameters

numRows     = 10;
numCols     = 5;
paramLambda = 0.5;
paramP      = rand([1, 1]) + 1;

difMode     = DIFF_MODE_CENTRAL;
epsVal      = 1e-6;

numIterations   = 25;


%% Generate Data

mA = randn([numRows, numCols]);
vB = randn([numRows, 1]);


%% Validate Derivative

vX          = randn([numCols, 1]);
hNormFun    = @(vX) sum(abs(vX) .^ paramP);

vGNumerical = CalcFunGrad(vX, hNormFun, difMode, epsVal);
% vGAnalytic  = paramP * vX .* (abs(vX) .^ (paramP - 2));

mX = diag(abs(vX) .^ (paramP - 2));
vGAnalytic = paramP * mX * vX;

disp(['Maximum Deviation Between Analytic and Numerical Derivative - ', num2str( max(abs(vGNumerical - vGAnalytic)) )]);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numCols)
    minimize( (0.5 * sum_square(mA * vX - vB)) + (paramLambda * pow_pos(norm(vX, paramP), paramP)) )
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Iterative Reweighted Least Squares (IRLS)

hObjFun = @(vX) (0.5 * sum((mA * vX - vB) .^ 2)) + (paramLambda * sum(abs(vX) .^ paramP));
vObjVal = zeros([numIterations, 1]);

mAA = mA.' * mA;
vAb = mA.' * vB;

vX          = mA \ vB; %<! Initialization by the Least Squares Solution
vObjVal(1)  = hObjFun(vX);

for ii = 2:numIterations
    
    mX = diag(abs(vX) .^ (paramP - 2));
    
    vX = (mAA + (paramLambda * paramP * mX)) \ vAb;
    
    vObjVal(ii) = hObjFun(vX);
end

disp([' ']);
disp(['Iterative Reweighted Least Squares (IRLS) Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjVal(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot(1:numIterations, [vObjVal, cvx_optval * ones([numIterations, 1])]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Objective Function Value vs. Iteration'], ['\lambda = ', num2str(paramLambda), ', p = ', num2str(paramP)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
set(hAxes, 'XLim', [1, numIterations]);
hLegend = ClickableLegend({['IRLS'], ['Optimal Value (CVX)']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

