% Mathematics Q3307741
% https://math.stackexchange.com/questions/3558240
% How to Solve Linear Least Squares Problem with Box Constraints
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     24/02/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

numRows = 20;
numCols = 15;

epsVal = 1e-6;

numIterations   = 200;
stopTol         = 0; %<! Run all iterations

solverIdx       = 0;
cMethodString   = {};

mObjFunVal  = zeros([numIterations, 1]);
mSolErrNorm = zeros([numIterations, 1]);


%% Load / Generate Data

mA = randn(numRows, numCols);
vB = randn(numRows, 1);

vC = 5 * (rand(numCols, 1) - 0.5);
vD = vC + (5 * rand(numCols, 1));

vX0 = mA \ vB;

hObjFun = @(vX) (0.5 * sum( (mA * vX - vB) .^ 2)) + (1e9 * any(vX < vC)) + (1e9 * any(vX > vD));


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vXCvx(numCols)
    minimize( (0.5 * sum_square(mA * vXCvx - vB)) )
    subject to
        vC <= vXCvx <= vD;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vXCvx.'), ' ]']);
disp([' ']);

sCvxSol.vXCvx     = vXCvx;
sCvxSol.cvxOptVal = cvx_optval;


%% Solution by Projected Gradient Method

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Projected Gradient Method'];

[vX, mX] = SolveLsBoxConstraints(mA, vB, vC, vD, vX0, numIterations, stopTol);

objVal = hObjFun(vX);

disp([' ']);
disp([cLegendString{solverIdx}, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(objVal)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

[mObjFunVal, mSolErrNorm] = UpdateAnalysisData(mObjFunVal, mSolErrNorm, mX, hObjFun, sCvxSol, solverIdx);


%% Solution by Accelerated Projected Gradient Method

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Accelerated Projected Gradient Method'];

[vX, mX] = SolveLsBoxConstraintsAccel(mA, vB, vC, vD, vX0, numIterations, stopTol);

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
hLineSeries = plot(1:numIterations, 10 * log10(mSolErrNorm));
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['Solution Error Norm'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', '$ 10 \log_{10} \left( {\left\| x - {x}_{CVX} \right\|}_{1} \right) $', ...
    'FontSize', fontSizeAxis, 'Interpreter', 'latex');
set(hAxes, 'XLim', [1, numIterations]);
hLegend = ClickableLegend(cLegendString);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

