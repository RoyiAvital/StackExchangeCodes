% Mathematics Q73712
% https://math.stackexchange.com/questions/73712
% Testing constrained linear least squares for optimality
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     04/09/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numRows     = 4;
numCols     = 7;

numRowsCons = 5;
numColsCons = numCols;

numIterations   = 25000;
stopThr         = 1e-6;
stepSize        = 4.5e-4;


%% Generate Data

mA = randn([numRows, numCols]);
vB = randn([numRows, 1]);

mC = randn([numRowsCons, numCols]);
vD = randn([numRowsCons, 1]);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numCols);
    minimize( 0.5 * sum_square(mA * vX - vB) );
    subject to
        mC * vX <= vD;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Projected Gradient Descent

hObjFun = @(vX) (0.5 * sum((mA * vX - vB) .^ 2));
vObjVal = zeros([numIterations, 1]);

mAA = mA.' * mA;
vAb = mA.' * vB;

vX          = zeros([numCols, 1]);
vObjVal(1)  = hObjFun(vX);

for ii = 2:numIterations
    
    vG = (mAA * vX) - vAb; %<! Gradient
    
    vX = vX - (stepSize * vG); %<! Gradient Descent
    vX = ProjectOntoLinearInequality(vX, mC, vD, stopThr); %<! Projection
    
    vObjVal(ii) = hObjFun(vX);
end

disp([' ']);
disp(['Projected Gradient Descent Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjVal(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot(1:numIterations, [vObjVal, cvx_optval * ones([numIterations, 1])]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', ['Objective Function Value vs. Iteration'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
set(hAxes, 'XLim', [1, numIterations]);
hLegend = ClickableLegend({['Projected Gradient Descent'], ['Optimal Value (CVX)']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot(1:numIterations, 10 * log10(abs(vObjVal - cvx_optval)));
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['Objective Function Value vs. CVX Reference'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value - CVX Reference [dB]', ...
    'FontSize', fontSizeAxis);
set(hAxes, 'XLim', [1, numIterations]);
% hLegend = ClickableLegend({['Projected Gradient Descent'], ['Optimal Value (CVX)']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

