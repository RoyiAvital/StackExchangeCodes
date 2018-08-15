% Mathematics Q1891878
% https://math.stackexchange.com/questions/1891878
% Least Squares with Quadratic (Positive Definite) Constraint
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     15/08/2018
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numRows = 6;
numCols = 5;

numIterations   = 100;
stepSize        = 0.005;


%% Generate Data

mA = randn(numRows * numRows, numCols * numCols);
vB = randn(numRows * numRows, 1);


%% CVX Solution

cvx_begin('quiet')
    cvx_precision('best');
    variable mX(numCols, numCols);
    minimize( 0.5 * pow_pos(norm( mA * vec(mX) - vB, 2 ), 2) );
    mX == semidefinite(numCols);
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mX(:).'), ' ]']);
disp([' ']);


%% Solution by Projected Gradient Descent

hObjFun = @(mX) (0.5 * (norm( (mA * mX(:)) - vB, 2 ) ^ 2));
vObjVal = zeros([numIterations, 1]);

mAA = mA.' * mA;
vAB = mA.' * vB;

vX         = mA \ vB; %<! Initialization by the Least Squares Solution
mX         = ProjectSymmetricMatrixSet(reshape(vX, numCols, numCols));
mX         = ProjectPsdMatrixSet(mX);
vX(:)      = mX(:);
vObjVal(1) = hObjFun(mX);

for ii = 2:numIterations
    
    vG = (mAA * vX) - vAB;
    vX = vX - (stepSize * vG);
    
    % Projection Step
    mX         = ProjectSymmetricMatrixSet(reshape(vX, numCols, numCols));
    mX         = ProjectPsdMatrixSet(mX);
    vX(:)       = mX(:);
    
    vObjVal(ii) = hObjFun(mX);
end

disp([' ']);
disp(['Projected Gradient Descent Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjVal(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mX(:).'), ' ]']);
disp([' ']);

figureIdx = figureIdx + 1;

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

