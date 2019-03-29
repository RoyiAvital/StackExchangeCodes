% Mathematics Q3079400
% https://math.stackexchange.com/questions/3079400
% Numerical Implementation: Solution for the Euler Lagrange Equation Of the Rudin Osher Fatemi (ROF) Total Variation Denoising Model
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     29/03/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

imageFileName   = 'Lena.png';
paramLambda     = 0.05;

stepSize        = 1 / (4 * paramLambda * paramLambda); %<! The Max Eigen Value of mD as Gradient is 1.
numIterations   = 600;


%% Load / Generate Data

mI = im2double(imread(imageFileName));
vF = mI(:);

numRows = size(mI, 1);
numCols = size(mI, 2);

mD = CreateGradientOperator(numRows, numCols);

hObjFun = @(vU) (0.5 * sum( (vU - vF) .^ 2)) + (paramLambda * norm(mD * vU, 1));


%% Solution by CVX

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vU(numRows * numCols);
    minimize( (0.5 * sum_square(vU - vF)) + (paramLambda * norm(mD * vU, 1)) );
cvx_end

toc();

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp([' ']);

vURef = vU;


%% Solution by Numerical Solver - Antonin Chambolle's Method
%{
Solving
%}

[vU, vObjFunVal] = TotalVariationDenoisingChambolle(vF, mD, paramLambda, stepSize, numIterations, hObjFun);

disp([' ']);
disp(['Projected Gradient Descent Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjFunVal(numIterations))]);
% disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp([' ']);


%% Display Results

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot([1:numIterations], [cvx_optval * ones([numIterations, 1]), vObjFunVal]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2:end), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', ['Least Squares with $ {L}_{1} $ Regularization - $ \arg \min_u \frac{1}{2} {\left\| u - f \right\|}_{2}^{2} + \lambda {\left\| D u \right\|}_{1} $'], ...
    'FontSize', fontSizeTitle, 'Interpreter', 'latex');
set(get(hAxes, 'XLabel'), 'String', 'Iteration Index', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['CVX'], ['Chambolle''s Method']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100, 100, 840, 300], 'Units', 'pixels');
hAxes       = axes(hFigure, 'Units', 'pixels', 'Position', [16, 20, 256, 256]);
hImgObj     = image(repmat(mI, [1, 1, 3]));
set(get(hAxes, 'Title'), 'String', ['Input Image - Lena'], ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTick', []);
set(hAxes, 'XTickLabel', []);
set(hAxes, 'YTick', []);
set(hAxes, 'YTickLabel', []);

hAxes       = axes(hFigure, 'Units', 'pixels', 'Position', [296, 20, 256, 256]);
hImgObj     = image(repmat(reshape(vURef, [numRows, numCols]), [1, 1, 3]));
set(get(hAxes, 'Title'), 'String', ['Output Image - CVX'], ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTick', []);
set(hAxes, 'XTickLabel', []);
set(hAxes, 'YTick', []);
set(hAxes, 'YTickLabel', []);

hAxes       = axes(hFigure, 'Units', 'pixels', 'Position', [574, 20, 256, 256]);
hImgObj     = image(repmat(reshape(vU, [numRows, numCols]), [1, 1, 3]));
set(get(hAxes, 'Title'), 'String', ['Output Image - Chambolle'], ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTick', []);
set(hAxes, 'XTickLabel', []);
set(hAxes, 'YTick', []);
set(hAxes, 'YTickLabel', []);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

