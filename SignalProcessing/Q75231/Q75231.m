% StackExchange Signal Processing Q75231
% https://dsp.stackexchange.com/questions/75231
% Image Denoising Using Total Variation (TV) Minimization with ADMM
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     18/05/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

CONV_SHAPE_FULL     = 1;
CONV_SHAPE_SAME     = 2;
CONV_SHAPE_VALID    = 3;


%% Simulation Parameters

imageFileName = 'Lena256.png';

paramLambda = 0.025; %<! TV Norm

numIterations   = 250;
noiseStd        = 10 / 255;


%% Generate Data

mI = im2double(imread(imageFileName));

numRows = size(mI, 1);
numCols = size(mI, 2);

mY = mI + (noiseStd * randn(numRows, numCols));
vY = mY(:);

vXInit  = zeros(numRows * numCols, 1);
% Generate the Diff Operator (2D Gradient) by Finite Differences
mD      = CreateGradientOperator(numRows, numCols);

% Objective Function
hObjFun = @(vX) (0.5 * sum( (vX - vY) .^ 2)) + (paramLambda * sum(abs(mD * vX)));

% Analysis
solverIdx       = 0;
cMethodString   = {};

mObjFunValMse   = zeros(numIterations, 1);
mSolMse         = zeros(numIterations, 1);


%% Display Data

figureIdx = figureIdx + 1;

hFigure = figure('Position', [100, 100, 542, 642]);
hAxes   = axes(hFigure, 'Units', 'pixels', 'Position', [010, 326, 256, 256]);
hImgObj = image(repmat(mI, [1, 1, 3]));
set(get(hAxes, 'Title'), 'String', {['Reference Image']}, ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTick', [], 'XTickLabel', []);
set(hAxes, 'YTick', [], 'YTickLabel', []);

hAxes   = axes(hFigure, 'Units', 'pixels', 'Position', [276, 326, 256, 256]);
hImgObj = image(repmat(mY, [1, 1, 3]));
set(get(hAxes, 'Title'), 'String', {['Input Image (Noisy)']}, ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTick', [], 'XTickLabel', []);
set(hAxes, 'YTick', [], 'YTickLabel', []);


%% Solution by CVX

solverString = 'CVX';

% cvx_solver('SDPT3'); %<! Default, Keep numRows low
% cvx_solver('SeDuMi');
cvx_solver('Mosek'); %<! Can handle numRows > 500, Very Good!
% cvx_solver('Gurobi');

hRunTime = tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numRows * numCols);
    minimize( (0.5 * sum_square(vX - vY)) + (paramLambda * norm(mD * vX, 1)) );
cvx_end

runTime = toc(hRunTime);

DisplayRunSummary(solverString, hObjFun, vX, runTime, cvx_status);

sCvxSol.vXCvx     = vX(:);
sCvxSol.cvxOptVal = hObjFun(vX);


%% Display Result

hAxes   = axes(hFigure, 'Units', 'pixels', 'Position', [010, 010, 256, 256]);
hImgObj = image(repmat(reshape(vX, numRows, numCols), [1, 1, 3]));
set(get(hAxes, 'Title'), 'String', {['Denoised Image - CVX']}, ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTick', [], 'XTickLabel', []);
set(hAxes, 'YTick', [], 'YTickLabel', []);


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


%% Display Results

hAxes   = axes(hFigure, 'Units', 'pixels', 'Position', [276, 010, 256, 256]);
hImgObj = image(repmat(reshape(vX, numRows, numCols), [1, 1, 3]));
set(get(hAxes, 'Title'), 'String', {['Denoised Image - ADMM']}, ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTick', [], 'XTickLabel', []);
set(hAxes, 'YTick', [], 'YTickLabel', []);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

figureIdx = figureIdx + 1;

hFigure = DisplayComparisonSummary(numIterations, mObjFunValMse, mSolMse, cLegendString, figPosLarge, lineWidthNormal, fontSizeTitle, fontSizeAxis);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

