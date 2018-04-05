% Mathematics Q1596362
% https://math.stackexchange.com/questions/1596362
% Linear Least Squares with Linear Equality Constraints - Iterative Solver
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     08/09/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;


%% Simulation Parameters

numRowsA = 10;
numColsA = 5;
numRowsB = 3;
numColsB = numColsA;

stepSize        = 0.005;
numIterations   = 250;


%% Generate Data

mA = randn([numRowsA, numColsA]);
vB = randn([numRowsA, 1]);

mB = randn([numRowsB, numColsB]);
vD = randn([numRowsB, 1]);

%% Solution by CVX - Problem II

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numColsA)
    minimize( 0.5 * sum_square(mA * vX - vB) )
    subject to
        mB * vX == vD;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Projected Gradient Descent

hObjFun     = @(vX) (0.5 * sum((mA * vX - vB) .^ 2));
hProjFun    = @(vY) vY - (mB.' * ((mB * mB.') \ (mB * vY - vD)));
vObjVal = zeros([numIterations, 1]);

mAA = mA.' * mA;
vAb = mA.' * vB;

vX          = mB \ vD; %<! Initialization by the Least Squares Solution of the Constraint
vX          = hProjFun(vX);
vObjVal(1)  = hObjFun(vX);

for ii = 2:numIterations
    
    vX = vX - (stepSize * ((mAA * vX) - vAb));
    vX = hProjFun(vX);
    
    vObjVal(ii) = hObjFun(vX);
end

disp([' ']);
disp(['Projected Gradient Descent Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjVal(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot(1:numIterations, [vObjVal, cvx_optval * ones([numIterations, 1])]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Objective Function Value vs. Iteration']}, ...
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


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

