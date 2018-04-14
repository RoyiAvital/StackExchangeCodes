% Mathematics Q1344369
% https://math.stackexchange.com/questions/1344369
% Compressive Sensing over the Complex Domain
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     14/04/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numRows = 64;
numCols = 256;

paramLambda = 4.0;

numIterations = 1000;

methodIdx = 0;


%% Generate Data

mA = randn([numRows, numCols]) + (1i * randn([numRows, numCols]));
vB = randn([numRows, 1]) + (1i * randn([numRows, 1]));

hCalcErrorNorm = @(mX, vXRef) sum((mX - vXRef) .* conj(mX - vXRef));
hCalcObjFunVal = @(mX) 0.5 * sum(((mA * mX) - vB) .* conj((mA * mX) - vB)) + paramLambda * sum(abs(mX), 1);


%% Solution by CVX

tic();
cvx_begin quiet
    variable vXCvx(numCols) complex
    minimize( (0.5 * sum_square_abs((mA * vXCvx) - vB)) + paramLambda * norm(vXCvx, 1) )
cvx_end
toc();

% disp([' ']);
% disp(['CVX Solution Summary']);
% disp(['The CVX Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
% disp(['The Optimal Argument Is Given By - [ ', num2str(vXCvx.'), ' ]']);
% disp([' ']);


%% Sub Gradient Method

methodIdx   = methodIdx + 1;
methodName  = ['Sub Gradient Method'];

hRunTime = tic();
[vX, mX] = SolveLsL1ComplexSubGrad(mA, vB, paramLambda, numIterations);
runTime = toc(hRunTime);

cMethodString{methodIdx}    = methodName;
vRunTime(methodIdx)         = runTime;
mErrorNorm(:, methodIdx)    = hCalcErrorNorm(mX, vXCvx);
mObjFunVal(:, methodIdx)    = hCalcObjFunVal(mX);


%% Sub Gradient Method - Real Domain

methodIdx   = methodIdx + 1;
methodName  = ['Sub Gradient Method - Real Domain'];

hRunTime = tic();
[vX, mX] = SolveLsL1ComplexRealSubGrad(mA, vB, paramLambda, numIterations);
runTime = toc(hRunTime);

cMethodString{methodIdx}    = methodName;
vRunTime(methodIdx)         = runTime;
mErrorNorm(:, methodIdx)    = hCalcErrorNorm(mX, vXCvx);
mObjFunVal(:, methodIdx)    = hCalcObjFunVal(mX);


%% Proximal Gradient Method

methodIdx   = methodIdx + 1;
methodName  = ['Proximal Gradient Method'];

hRunTime = tic();
[vX, mX] = SolveLsL1ComplexPgm(mA, vB, paramLambda, numIterations);
runTime = toc(hRunTime);

cMethodString{methodIdx}    = methodName;
vRunTime(methodIdx)         = runTime;
mErrorNorm(:, methodIdx)    = hCalcErrorNorm(mX, vXCvx);
mObjFunVal(:, methodIdx)    = hCalcObjFunVal(mX);


%% Proximal Gradient Method - Real Domain

methodIdx   = methodIdx + 1;
methodName  = ['Proximal Gradient Method - Real Domain'];

hRunTime = tic();
[vX, mX] = SolveLsL1ComplexRealPgm(mA, vB, paramLambda, numIterations);
runTime = toc(hRunTime);

cMethodString{methodIdx}    = methodName;
vRunTime(methodIdx)         = runTime;
mErrorNorm(:, methodIdx)    = hCalcErrorNorm(mX, vXCvx);
mObjFunVal(:, methodIdx)    = hCalcObjFunVal(mX);


%% Alternating Direction Method of Multipliers (ADMM) Method

methodIdx   = methodIdx + 1;
methodName  = ['ADMM Method'];

hRunTime = tic();
[vX, mX] = SolveLsL1ComplexAdmm(mA, vB, paramLambda, numIterations);
% [vX, mX] = SolveLsL1Admm(mA, vB, paramLambda, numIterations);
runTime = toc(hRunTime);

cMethodString{methodIdx}    = methodName;
vRunTime(methodIdx)         = runTime;
mErrorNorm(:, methodIdx)    = hCalcErrorNorm(mX, vXCvx);
mObjFunVal(:, methodIdx)    = hCalcObjFunVal(mX);


%% Fixed Point Iteration Method (IRLS)

methodIdx   = methodIdx + 1;
methodName  = ['Fixed Point Iteration (IRLS) Method'];

hRunTime = tic();
[vX, mX] = SolveLsL1ComplexIrls(mA, vB, paramLambda, numIterations);
runTime = toc(hRunTime);

cMethodString{methodIdx}    = methodName;
vRunTime(methodIdx)         = runTime;
mErrorNorm(:, methodIdx)    = hCalcErrorNorm(mX, vXCvx);
mObjFunVal(:, methodIdx)    = hCalcObjFunVal(mX);


%% Display Results

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries = plot([1:numIterations], 10 * log10(mErrorNorm));
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['The Error Norm - $ {\left\| {x}^{k} - {x}_{CVX} \right\|}_{2}^{2} $'], ...
    'FontSize', fontSizeTitle, 'Interpreter', 'latex');
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Error Norm [dB]', ...
    'FontSize', fontSizeAxis);
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);
hLegend = ClickableLegend(cMethodString);
set(hLegend, 'FontSize', fontSizeAxis);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries = plot([1:numIterations], mObjFunVal);
set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(hLineSeries(5), 'LineStyle', ':', 'LineWidth', lineWidthThin);
set(get(hAxes, 'Title'), 'String', ['The Objection Function Value - $ \frac{1}{2} {\left\| A {x}^{k} - b \right\|}_{2}^{2} + \lambda {\left\| {x}^{k} \right\|}_{1} $'], ...
    'FontSize', fontSizeTitle, 'Interpreter', 'latex');
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);
hLegend = ClickableLegend(cMethodString);
set(hLegend, 'FontSize', fontSizeAxis);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hBarObj = bar(1:length(vRunTime), vRunTime);
% set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(hLineSeries(2:end), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', ['L_1 Regularized Least Squares - Methods Run Time'], ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTickLabel', cMethodString, 'XTickLabelRotation', 45);
set(get(hAxes, 'XLabel'), 'String', 'Method', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Run Time [Sec]', ...
    'FontSize', fontSizeAxis);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

