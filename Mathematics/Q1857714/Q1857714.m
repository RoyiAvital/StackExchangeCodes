% Mathematics Q1857714
% https://math.stackexchange.com/questions/1857714
% Optimization of Convex Function with L∞L∞ Norm Regularization (Composition Model)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     23/08/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numRows = 4;
numCols = 3; %<! Number of Vectors - i (K in the question)

paramLambda = 0.75;

numIterations   = 250;


%% Generate Data

mA = randn([numRows, numCols]);
vB = randn([numRows, 1]);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numCols)
    minimize( 0.5 * square_pos(  norm(mA * vX - vB, 2) ) + (paramLambda * norm(vX, inf)) );
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Proximal Gradient Descent

hObjFun = @(vX) (0.5 * sum((mA * vX - vB) .^ 2)) + (paramLambda * norm(vX, 'inf'));

[vX, mX] = SolveLsLInfProx(mA, vB, paramLambda, numIterations);

objVal = hObjFun(vX);

disp([' ']);
disp(['Projected Gradient Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(objVal)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Display Results

vObjVal = zeros([numIterations, 1]);

for ii = 1:numIterations
    vObjVal(ii) = hObjFun(mX(:, ii));
end

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot(1:numIterations, [vObjVal, cvx_optval * ones([numIterations, 1])]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', ['Objective Function Value vs. Iteration'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
set(hAxes, 'XLim', [1, numIterations]);
hLegend = ClickableLegend({['PGM'], ['Optimal Value (CVX)']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

