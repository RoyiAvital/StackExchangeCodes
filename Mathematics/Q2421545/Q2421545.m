% Mathematics Q2421545
% https://math.stackexchange.com/questions/2421545
% Least Square with Optimization of Diagonal Matrix
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     09/09/2017
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

numRowsA    = 10;
numColsA    = 6;

numRowsB    = 10;
numColsB    = numColsA;

numRowsC = numRowsA;
numColsC = numRowsB;

difMode     = DIFF_MODE_CENTRAL;
epsVal      = 1e-6;

stepSize        = 0.005;
numIterations   = 250;


%% Generate Data

mA = randn([numRowsA, numColsA]);
mB = randn([numRowsB, numColsB]);
mC = randn([numRowsC, numColsC]);


%% Projection onto Diagonal Matrices Set

mY = randn([numRowsA, numRowsA]);

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numRowsA)
    minimize( norm(diag(vX) - mY, 'fro') )
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

disp([' ']);
disp(['Projection Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(norm(diag(diag(mY)) - mY, 'fro'))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(diag(mY).'), ' ]']);
disp([' ']);


%% Validate Derivative of \frac{1}{2} \left\| sum_{i} {s}_{i} {a}_{i} {b}_{i}^{T} - C \right\|_{F}^{2}

vX          = randn([numColsA, 1]);
hNormFun    = @(vX) 0.5 * (norm( (mA * diag(vX) * mB.') - mC, 'fro' ) ^ 2);

vGNumerical = CalcFunGrad(vX, hNormFun, difMode, epsVal);

vGAnalytic = zeros([numColsA, 1]);

for ii = 1:numColsA
    vGAnalytic(ii) = mA(:, ii).' * ((mA * diag(vX) * mB.') - mC) * mB(:, ii);
end

disp(['Maximum Deviation Between Analytic and Numerical Derivative - ', num2str( max(abs(vGNumerical - vGAnalytic)) )]);


%% Solution by CVX - Problem II

cvx_begin('quiet')
    cvx_precision('best');
    variable vS(numColsA);
    minimize( 0.5 * pow_pos(norm( (mA * diag(vS) * mB.') - mC, 'fro' ), 2) );
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vS.'), ' ]']);
disp([' ']);


%% Solution by Projected Gradient Descent

hObjFun = @(vX) (0.5 * (norm( (mA * diag(vX) * mB.') - mC, 'fro' ) ^ 2));
vObjVal = zeros([numIterations, 1]);

mAA     = mA.' * mA;
mBB     = mB.' * mB;
mAyb    = mA.' * mC * mB;

mS          = mAA \ (mA.' * mC * mB) / mBB; %<! Initialization by the Least Squares Solution
vS          = diag(mS);
mS          = diag(vS);
vObjVal(1)  = hObjFun(vS);

for ii = 2:numIterations
    
    mG = (mAA * mS * mBB) - mAyb;
    mS = mS - (stepSize * mG);
    
    % Projection Step
    vS          = diag(mS);
    mS          = diag(vS);
    
    vObjVal(ii) = hObjFun(vS);
end

disp([' ']);
disp(['Projected Gradient Descent Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjVal(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vS.'), ' ]']);
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


%% Solution by Gradient Descent

vObjVal = zeros([numIterations, 1]);

mS          = mAA \ (mA.' * mC * mB) / mBB; %<! Initialization by the Least Squares Solution
vS          = diag(mS);
vObjVal(1)  = hObjFun(vS);

vG = zeros([numColsA, 1]);

for ii = 2:numIterations
    
    for jj = 1:numColsA
        vG(jj) = mA(:, jj).' * ((mA * diag(vS) * mB.') - mC) * mB(:, jj);
    end
    
    vS = vS - (stepSize * vG);
    
    vObjVal(ii) = hObjFun(vS);
end

disp([' ']);
disp(['Gradient Descent Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjVal(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vS.'), ' ]']);
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
hLegend = ClickableLegend({['Gradient Descent'], ['Optimal Value (CVX)']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

