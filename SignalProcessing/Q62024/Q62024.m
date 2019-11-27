% StackExchange Signal Processing Q62024
% https://dsp.stackexchange.com/questions/62024
% Proximal Gradient Method (PGM) for a Function Model with More than 2 Functions (Sum of Functions)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     25/11/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numElements = 40;
paramLambda1 = 0.5; %<! L1 Norm
paramLambda2 = 0.75; %<! TV Norm

numIterations   = 50000;
stepSize        = 0.00015;

cMethodNames = {['CVX'], ['Sub Gradient Method']};
% cMethodNames = {['CVX'], ['Sub Gradient Method'], ['Proximal Gradient Method'], ['ADMM']};
methodIdx = 0;


%% Generate Data

vY = 10 * randn(numElements, 1);

% Generate the Diff Operator (1D Gradient) by Finite Differences
mD = spdiags([-ones(numElements, 1), ones(numElements, 1)], [0, 1], numElements - 1, numElements);

% Objective Function
hObjFun = @(vX) (0.5 * sum( (vX - vY) .^ 2)) + (paramLambda1 * sum(abs(vX))) + (paramLambda2 * sum(abs(mD * vX)));

numMethods  = size(cMethodNames, 1);
mObjVal     = zeros(numIterations, numMethods);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numElements);
    minimize( (0.5 * pow_pos(norm(vX - vY, 2), 2)) + (paramLambda1 * norm(vX, 1)) + (paramLambda2 * norm(mD * vX, 1)));
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp([' ']);

methodIdx = methodIdx + 1;
mObjVal(:, methodIdx) =  cvx_optval * ones([numIterations, 1]);



%% Solution by Sub Gradient Descent
%{
Solving $ \arg \min_x \frac{1}{2} {\left\| x - y \right\|}_{2}^{2} +
{\lambda}_{1} {\left\| x \right\|}_{1} + {\lambda}_{2} {\left\| D x \right\|}_{1} $
%}

methodIdx = methodIdx + 1;

vX = vY;
vG = vX;
mObjVal(1, methodIdx) = hObjFun(vX);

for ii = 2:numIterations
    vG(:) = (vX - vY) + (paramLambda1 * sign(vX)) + (paramLambda2 * mD.' * sign(mD * vX));
    vX(:) = vX - (stepSize * vG);
    mObjVal(ii, methodIdx) = hObjFun(vX);
end

disp([' ']);
disp(['Projected Gradient Descent Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp([' ']);


%% Solution by Prox Function (Closed Form Solution)
%{
Solving
%}

% vX = ProjectWeightedL2Ball(vY, mW, vC, paramLambda);
% 
% disp([' ']);
% disp(['Projected Gradient Descent Solution Summary']);
% disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
% disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
% disp([' ']);


%% Display Results

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot([1:numIterations], mObjVal);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2:end), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Least Squares with {L}_{1} (LASSO) and Total Varitaion Norm Regularization'], ['SubStream - ', num2str(subStreamNumber)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Index', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend(cMethodNames);
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

