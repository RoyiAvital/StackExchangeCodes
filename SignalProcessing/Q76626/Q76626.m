% StackExchange Signal Processing Q76626
% https://dsp.stackexchange.com/questions/76626
% Solve Efficiently the 1D L1 Regularized Least Squares Problem (Denoising / Deblurring)
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

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Simulation Parameters

% Signal Parameters
numSamples  = 1000;
numNnzFctr  = 0.1;
noiseStd    = 0.025;

% Filter Paramertes
numCoef     = 11;
convShape   = CONVOLUTION_SHAPE_VALID;

% Model Parameters
paramLambda = 0.15; %<! L1 Norm

% Solver Parameters
numIterations = 5000;


%% Generate Data

vA = rand(numCoef, 1);
vA = vA / sum(vA);

mA      = CreateConvMtx1D(vA, numSamples, convShape);
numNnz  = round(numNnzFctr * numSamples);
vI      = randperm(numSamples, numNnz);
vW      = zeros(numSamples, 1);
vW(vI)  = randi([5, 10], numNnz, 1);
vY      = (mA * vW) + (noiseStd * randn(size(mA, 1), 1));

vXInit = zeros(size(mA, 2), 1);

% Objective Function
hObjFun = @(vX) (0.5 * sum( (mA * vX - vY) .^ 2)) + (paramLambda * sum(abs(vX)));

% Analysis
solverIdx       = 0;
cMethodString   = {};

mObjFunValMse   = zeros(numIterations, 1);
mSolMse         = zeros(numIterations, 1);


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
    minimize( (0.5 * sum_square(mA * vX - vY)) + (paramLambda * norm(vX, 1)) );
cvx_end

runTime = toc(hRunTime);

DisplayRunSummary(solverString, hObjFun, vX, runTime, cvx_status);

sCvxSol.vXCvx     = vX(:);
sCvxSol.cvxOptVal = hObjFun(vX);


%% Solution by ADMM
%{
Solving:

$$ \arg \min_{ x \in \mathbb{R}^{n} } \frac{1}{2} {\left\| A x - y \right|}_{2}^{2} + \lambda {\left\| x \right\|}_{1} $$
%}

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['ADMM'];

hRunTime = tic();

[vX, mX] = SolveLsL1Admm(vXInit, mA, vY, paramLambda, 'numIterations', numIterations);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFun, vX, runTime);

[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFun, sCvxSol, solverIdx);


%% Solution by Majorization Minimization
%{
Solving:

$$ \arg \min_{ x \in \mathbb{R}^{n} } \frac{1}{2} {\left\| A x - y \right|}_{2}^{2} + \lambda {\left\| x \right\|}_{1} $$
%}

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['MM'];

hRunTime = tic();

[vX, mX] = SolveLsL1Mm(vXInit, mA, vY, paramLambda, 'numIterations', numIterations);

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
hLineObj = plot(1:numSamples, [vW, vX]);
set(hLineObj(1), 'LineStyle', 'none', 'Marker', '*');
set(hLineObj(2), 'LineStyle', 'none', 'Marker', 'x');
set(get(hAxes, 'Title'), 'String', {['Signals']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['L1 Estimation']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

