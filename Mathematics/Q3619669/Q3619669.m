% Mathematics Q3619669
% https://math.stackexchange.com/questions/3619669
% Variation of Least Squares with Symmetric Positive Semi Definite (PSD)
% Constraint
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     12/04/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 179;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

numRows     = 4;
numVectors  = 4;

epsVal = 1e-6;

numIterations   = 20000;
stopTol         = 0; %<! Run all iterations

solverIdx       = 0;
cMethodString   = {};

mObjFunVal  = zeros([numIterations, 1]);
mSolErrNorm = zeros([numIterations, 1]);


%% Load / Generate Data

mXX = randn(numRows, numVectors); %<! The set of {x}_{i} (Each column)

mX = zeros(numVectors, numRows * numRows);
for ii = 1:numVectors
    mX(ii, :) = kron(mXX(:, ii).', mXX(:, ii).');
end

vY = randn(numVectors, 1);

mW0 = eye(numRows);

hObjFun = @(mW) 0.5 * sum((mX * mW(:) - vY) .^ 2);


%% Solution by CVX

cvx_solver('SDPT3'); %<! Default, Slowest
% cvx_solver('Mosek'); %<! Fastest, Requires removing 'cvx_precision('best');'
% cvx_solver('SeDuMi'); %<! Faster than 'SDPT3', yet less accurate

hRunTime = tic();

cvx_begin('quiet')
    % cvx_precision('best'); %<! Makes Mosek Sovler fail
    variable mW(numRows, numRows) semidefinite
    objVal = 0;
    for ii = 1:numVectors
        objVal = objVal + square( (mXX(:, ii).' * mW * mXX(:, ii)) - vY(ii) );
    end
    minimize( 0.5 * objVal )
cvx_end

runTime = toc(hRunTime);

% hRunTime = tic();
% 
% cvx_begin('quiet')
%     % cvx_precision('best');
%     variable mW(numRows, numRows) semidefinite
%     minimize( 0.5 * sum_square( mX * vec(mW) - vY ) )
% cvx_end
% 
% runTime = toc(hRunTime);

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(mW))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mW(:).'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);

sCvxSol.vXCvx     = mW(:);
sCvxSol.cvxOptVal = cvx_optval;
sCvxSol.cvxOptVal = hObjFun(mW);


%% Solution by Projected Gradient Descent Method

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Projected Gradient Descent Method'];

hRunTime = tic();
[mW, mX] = SolveLsPosSemiDefinite(mX, vY, mW0(:), numIterations, stopTol);
runTime = toc(hRunTime);

mW = reshape(mW, numRows, numRows);

objVal = hObjFun(mW);

disp([' ']);
disp([cLegendString{solverIdx}, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(objVal)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mW(:).'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
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

