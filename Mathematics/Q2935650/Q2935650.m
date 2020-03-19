% Mathematics Q2935650
% https://math.stackexchange.com/questions/2935650
% Solve Linear Least Squares Problem with Unit Simplex Constraint
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     02/03/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 2090; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

OPERATION_MODE_ALTERNATING_PROJECTIONS  = 1;
OPERATION_MODE_DIRECT_PROJECTION        = 2;


%% Parameters

numRows = 1500;
numCols = 500;


%% Load / Generate Data

mA = randn(numRows, numCols);
vB = randn(numRows, 1);

hObjFun = @(vX) (0.5 * sum((mA * vX - vB) .^ 2)) + (1e9 * (any(vX < 0))) + (1e9 * (abs(sum(vX) - 1) > 1e-5));

vX0 = (1 / numCols) * rand(numCols, 1);
vX0 = vX0 - ((sum(vX0) - 1) / numCols);

numIterations   = 2000;
stopTol         = 0; %<! Run all iterations

solverIdx       = 0;
cMethodString   = {};

mObjFunVal  = zeros([numIterations, 1]);
mSolErrNorm = zeros([numIterations, 1]);


%% Solution by CVX

solverString = 'CVX';

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numCols, 1);
    % minimize( 0.5 * sum_square( mA * vX - vB ) );
    minimize( norm( mA * vX - vB ) );
    subject to
        vX >= 0;
        sum(vX) == 1;
cvx_end

toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

sCvxSol.vXCvx     = vX;
sCvxSol.cvxOptVal = hObjFun(vX);


%% Solution by Projected Gradient Descent (Alternating Projections)

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Projected Gradient Method (Alternating Projections)'];

tic();

[vX, mX] = SolveLsUnitSimplexProjectedGd(mA, vB, vX0, numIterations, stopTol, OPERATION_MODE_ALTERNATING_PROJECTIONS);

toc();

objVal = hObjFun(vX);

disp([' ']);
disp([cLegendString{solverIdx}, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(objVal)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

[mObjFunVal, mSolErrNorm] = UpdateAnalysisData(mObjFunVal, mSolErrNorm, mX, hObjFun, sCvxSol, solverIdx);


%% Solution by Projected Gradient Descent

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Projected Gradient Method'];

tic();

[vX, mX] = SolveLsUnitSimplexProjectedGd(mA, vB, vX0, numIterations, stopTol, OPERATION_MODE_DIRECT_PROJECTION);

toc();

objVal = hObjFun(vX);

disp([' ']);
disp([cLegendString{solverIdx}, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(objVal)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

[mObjFunVal, mSolErrNorm] = UpdateAnalysisData(mObjFunVal, mSolErrNorm, mX, hObjFun, sCvxSol, solverIdx);


% Solution by Conditional Gradient Method (Frank Wolfe Algorithms)

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Conditional Gradient Method'];

tic();

[vX, mX] = SolveLsUnitSimplexCgd(mA, vB, vX0, numIterations, stopTol);

toc();

objVal = hObjFun(vX);

disp([' ']);
disp([cLegendString{solverIdx}, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(objVal)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

[mObjFunVal, mSolErrNorm] = UpdateAnalysisData(mObjFunVal, mSolErrNorm, mX, hObjFun, sCvxSol, solverIdx);


%% Display Results

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);

hAxes       = subplot(2, 1, 1);
hLineSeries = plot(1:numIterations, 10 * log10(mObjFunVal));
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['Objective Function Value vs. Optimal Value (CVX)'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', '$ 10 \log_{10} \left( \left| f \left( x \right) - f \left( {x}_{CVX} \right) \right| \right) $', ...
    'FontSize', fontSizeAxis, 'Interpreter', 'latex');
set(hAxes, 'XLim', [1, numIterations]);
hLegend = ClickableLegend(cLegendString);

hAxes       = subplot(2, 1, 2);
hLineSeries = plot(1:numIterations, 20 * log10(mSolErrNorm));
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['Solution Error Norm'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', '$ 20 \log_{10} \left( {\left\| x - {x}_{CVX} \right\|}_{2} \right) $', ...
    'FontSize', fontSizeAxis, 'Interpreter', 'latex');
set(hAxes, 'XLim', [1, numIterations]);
hLegend = ClickableLegend(cLegendString);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

