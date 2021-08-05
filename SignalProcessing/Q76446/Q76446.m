% StackExchange Signal Processing Q76446
% https://dsp.stackexchange.com/questions/76446
% Solve Efficiently the 1D Total Variation Regularized Least Squares Problem (Denoising / Deblurring)
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     05/08/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numSections         = 20;
numSamplesSections  = 50;
paramLambda         = 0.25; %<! TV Norm
noiseStd            = 0.05;

% Solver Parameters

numIterations = 100;


%% Generate Data

numSamples = numSections * numSamplesSections;

% Generate the Diff Operator (1D Gradient) by Finite Differences
mD = spdiags([-ones(numSamples, 1), ones(numSamples, 1)], [0, 1], numSamples - 1, numSamples);
vW = reshape(repmat(randn(1, numSections), numSamplesSections, 1), numSamples, 1);
vY = vW + (noiseStd * randn(numSamples, 1));

vXInit = vY;

% Objective Function
hObjFun = @(vX) (0.5 * sum( (vX - vY) .^ 2)) + (paramLambda * sum(abs(mD * vX)));

% Analysis
solverIdx       = 0;
cMethodString   = {};

mObjFunValMse   = zeros(numIterations, 1);
mSolMse         = zeros(numIterations, 1);


%% Display Data

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
hLineObj = plot(1:numSamples, [vW, vY]);
set(hLineObj(1), 'LineWidth', lineWidthNormal);
set(hLineObj(2), 'LineStyle', 'none', 'Marker', '*');
set(get(hAxes, 'Title'), 'String', {['Input Signals']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['Input Noisy Samples']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Solution by CVX

solverString = 'CVX';

% cvx_solver('SDPT3'); %<! Default, Keep numRows low
% cvx_solver('SeDuMi');
% cvx_solver('Mosek'); %<! Can handle numRows > 500, Very Good!
% cvx_solver('Gurobi');

hRunTime = tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numSamples);
    minimize( (0.5 * sum_square(vX - vY)) + (paramLambda * norm(mD * vX, 1)) );
cvx_end

runTime = toc(hRunTime);

DisplayRunSummary(solverString, hObjFun, vX, runTime, cvx_status);

sCvxSol.vXCvx     = vX(:);
sCvxSol.cvxOptVal = hObjFun(vX);


%% Solution by ADMM
%{
Solving:

$$ \arg \min_{ x \in \mathbb{R}^{n} } \frac{1}{2} {\left\| x - y \right|}_{2}^{2} + \lambda {\left\| D x \right\|}_{1} $$
%}

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['ADMM'];

hRunTime = tic();

[vX, mX] = SolveProxTvAdmm(vXInit, vY, mD, paramLambda, 'numIterations', numIterations);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFun, vX, runTime);

[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFun, sCvxSol, solverIdx);


%% Solution by Majorization Minimization
%{
Solving:

$$ \arg \min_{ x \in \mathbb{R}^{n} } \frac{1}{2} {\left\| x - y \right|}_{2}^{2} + \lambda {\left\| D x \right\|}_{1} $$
%}

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['MM'];

hRunTime = tic();

[vX, mX] = SolveProxTvMm(vXInit, vY, mD, paramLambda, 'numIterations', numIterations);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFun, vX, runTime);

[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFun, sCvxSol, solverIdx);


%% Display Results

figureIdx = figureIdx + 1;

hFigure = DisplayComparisonSummary(numIterations, mObjFunValMse, mSolMse, cLegendString, figPosLarge, lineWidthNormal, fontSizeTitle, fontSizeAxis);

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
hLineObj = plot(1:numSamples, [vW, vY, vX]);
set(hLineObj(1), 'LineWidth', lineWidthNormal);
set(hLineObj(2), 'LineStyle', 'none', 'Marker', '*');
% set(hLineObj(3), 'LineWidth', lineWidthThin, 'LineStyle', ':');
set(hLineObj(3), 'LineStyle', 'none', 'Marker', 'x');
set(get(hAxes, 'Title'), 'String', {['Signals']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['Input Noisy Samples'], ['TV Estimation']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

