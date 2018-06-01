% Mathematics Q330718
% https://math.stackexchange.com/questions/330718
% Least Squares with Linear Inequality Constraints (Positivity) and Non Linear Equality of L2 Norm
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     31/05/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numRows         = 20;
numCols         = 10;

stepSize        = 1e-5;
numIterations   = 25000;

simplexBallRadius   = 1; %<! Unit Simplex Ball
stopThr             = 1e-2; %<! Projection onto Unit Simplex Ball


%% Generate Data

mA = randn([numRows, numCols]);
vB = randn([numRows, 1]);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numCols)
    minimize( (0.5 * sum_square((mA * vX) - vB )) )
    subject to
        vX >= 0;
        sum(vX) == 1;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Projected Sub Gradient

hNonNegativeConsFun = @(vX) 1e9 * any(vX < 0);
hL1NormConsFun      = @(vX) 1e9 * any(abs(norm(vX, 1) - 1) > stopThr);


hObjFun     = @(vX) (0.5 * sum(((mA * vX) - vB) .^ 2)) + hNonNegativeConsFun(vX) + hL1NormConsFun(vX);
hProjFun    = @(vX) ProjectSimplex(vX, simplexBallRadius, stopThr);

vObjVal = zeros([numIterations, 1]);

% First Iteration
vX          = mA \ vB;
vX          = hProjFun(vX);
vObjVal(1)  = hObjFun(vX);

mAA = mA.' * mA;
vAb = mA.' * vB;

for ii = 2:numIterations
    
    vG = (mAA * vX) - vAb;
    vX = vX - (stepSize * vG);
    vX = hProjFun(vX);
    
    vObjVal(ii) = hObjFun(vX);
end

disp([' ']);
disp(['Projected Sub Gradient Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjVal(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

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
hLegend = ClickableLegend({['Projected Sub Gradient'], ['Optimal Value (CVX)']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

