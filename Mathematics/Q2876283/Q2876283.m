% Mathematics Q2876283
% https://math.stackexchange.com/questions/2876283
% Least Square with Optimization of Triangular Matrix
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     12/08/2018
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numRowsA    = 6;
numColsA    = 20;

numRowsB    = 10;
numColsB    = numColsA;

stepSize        = 0.0025;
numIterations   = 500;


%% Generate Data

mA = randn([numRowsA, numColsA]);
mB = randn([numRowsB, numColsB]);


%% Projection onto Lower Triangular Matrices Set

mY = randn([numRowsA, numRowsA]);

cvx_begin('quiet')
    cvx_precision('best');
    variable mX(numRowsA, numRowsA) lower triangular
    minimize( norm(mX - mY, 'fro') )
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mX(:).'), ' ]']);
disp([' ']);

mX = tril(mY);

disp([' ']);
disp(['Projection Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(norm(mX - mY, 'fro'))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mX(:).'), ' ]']);
disp([' ']);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable mX(numRowsA, numRowsB) lower triangular;
    minimize( 0.5 * pow_pos(norm( mX * mB - mA, 'fro' ), 2) );
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mX(:).'), ' ]']);
disp([' ']);


%% Solution by Projected Gradient Descent

hObjFun = @(mX) (0.5 * (norm( (mX * mB) - mA, 'fro' ) ^ 2));
vObjVal = zeros([numIterations, 1]);

mBB = mB * mB.';
mAB = mA * mB.';

mX         = mAB * pinv(mBB); %<! Initialization by the Least Squares Solution
mX         = tril(mX);
vObjVal(1) = hObjFun(mX);

for ii = 2:numIterations
    
    mG = (mX * mBB) - mAB;
    mX = mX - (stepSize * mG);
    
    % Projection Step
    mX = tril(mX);
    
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

