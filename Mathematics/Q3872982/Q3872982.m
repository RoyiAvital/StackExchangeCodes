% StackExchange Mathematics Q3872982
% https://math.stackexchange.com/questions/3872982
% Solve Linear Least Squares with Squared L2 Norm Regularization (Tikhonov
% / Ridge Regression) with Non Negativity Constraint Using FISTA
% References:
%   1.  
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     26/12/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 0;2165;42; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

numRows         = 100;
numCols         = 20;
paramLambda     = 0.1;

% Solvers parameters
numIterations       = 5000;
stepSizeGd          = 5e-5;
stepSizeMomentum    = 0.8;
stepSizeAccel       = 0.4;


%% Load / Generate Data

mA = randn(numRows, numCols);
vB = randn(numRows, 1);

hObjFun = @(vX) 0.5 * sum((mA * vX - vB) .^ 2) + (0.5 * paramLambda * sum(vX .^ 2));

mAA = mA.' * mA;
vAb = mA.' * vB;

hG = @(vX) (mAA * vX) - vAb + (paramLambda * vX); %<! Gradient of the Objection Function
hP = @(vX) max(vX, 0); %<! Projection onto the constraint set


solverIdx       = 0;
cMethodString   = {};

mObjFunValMse   = zeros([numIterations, 1]);
mSolMse         = zeros([numIterations, 1]);


%% Solution by CVX

solverString = 'CVX';

% cvx_solver('SDPT3'); %<! Default, Keep numRows low
% cvx_solver('SeDuMi');
% cvx_solver('Mosek'); %<! Can handle numRows > 500, Very Good!
% cvx_solver('Gurobi');

hRunTime = tic();

cvx_begin('quiet')
% cvx_begin()
    % cvx_precision('best');
    variable vX(numCols, 1);
    minimize( 0.5 * sum_square(mA * vX - vB) + (0.5 * paramLambda * square_pos(norm(vX))) );
    subject to
        vX >= 0;
cvx_end

runTime = toc(hRunTime);

% vX = mX(:);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);

sCvxSol.vXCvx     = vX;
sCvxSol.cvxOptVal = hObjFun(vX);


%% Solution by Projected Gradient Descent

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Projected Gradient Method'];

hRunTime = tic();

[vX, mX] = ProjectedGd(zeros(numCols, 1), hG, hP, numIterations, stepSizeGd);

runTime = toc(hRunTime);

disp([' ']);
disp([cLegendString{solverIdx}, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);

[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFun, sCvxSol, solverIdx);


%% Solution by Projected Gradient Descent with Momentum

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Projected Gradient Descent with Momentum'];

hRunTime = tic();

[vX, ~, mX] = ProjectedGdMomentum(zeros(numCols, 1), zeros(numCols, 1), hG, hP, numIterations, stepSizeGd, stepSizeMomentum);

runTime = toc(hRunTime);

disp([' ']);
disp([cLegendString{solverIdx}, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);

[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFun, sCvxSol, solverIdx);


%% Solution by Projected Gradient Descent with Nesterov Acceleration

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Projected Gradient Descent with Nesterov Acceleration'];

hRunTime = tic();

[vX, ~, mX] = ProjectedGdAccel(zeros(numCols, 1), zeros(numCols, 1), hG, hP, numIterations, stepSizeGd, stepSizeAccel);

runTime = toc(hRunTime);

disp([' ']);
disp([cLegendString{solverIdx}, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);

[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFun, sCvxSol, solverIdx);


%% Solution by Projected Gradient Descent with FISTA

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Projected Gradient Descent with FISTA Acceleration'];

hRunTime = tic();

[vX, ~, mX] = ProjectedGdFista(zeros(numCols, 1), zeros(numCols, 1), hG, hP, numIterations, stepSizeGd);
% [vX, mX] = SolveLsFista(zeros(numCols, 1), mA, vB, paramLambda, numIterations, stepSizeGd);

runTime = toc(hRunTime);

disp([' ']);
disp([cLegendString{solverIdx}, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);

[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFun, sCvxSol, solverIdx);


%% Display Results

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);

hAxes       = subplot(2, 1, 1);
hLineSeries = plot(1:numIterations, 10 * log10(mObjFunValMse));
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['Objective Function Value vs. Optimal Value (CVX)'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', '$ 10 \log_{10} {\left( \left| f \left( x \right) - f \left( {x}_{CVX} \right) \right| \right)}^{2} $', ...
    'FontSize', fontSizeAxis, 'Interpreter', 'latex');
set(hAxes, 'XLim', [1, numIterations]);
hLegend = ClickableLegend(cLegendString);

hAxes       = subplot(2, 1, 2);
hLineSeries = plot(1:numIterations, 10 * log10(mSolMse));
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['Solution Error Norm'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', '$ 10 \log_{10} \left( {\left\| x - {x}_{CVX} \right\|}_{2}^{2} \right) $', ...
    'FontSize', fontSizeAxis, 'Interpreter', 'latex');
set(hAxes, 'XLim', [1, numIterations]);
hLegend = ClickableLegend(cLegendString);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

