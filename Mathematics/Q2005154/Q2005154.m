% Mathematics Q2005154
% https://math.stackexchange.com/questions/2005154
% How to Project onto the Intersection of Two Sets (Which Create the Unit Simplex) when Optimizing a Convex Function?
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     24/08/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numRows = 4;
numCols = 3; %<! Number of Vectors - i (K in the question)

numIterations   = 25;
stepSizeBase    = 0.25;

simplexRadius = 1;
stopThr = 1e-6; %<! Unit Simplex Projection


%% Generate Data

mA = randn([numRows, numCols]);
vB = randn([numRows, 1]);

hObjFun = @(vX) (0.5 * sum((mA * vX - vB) .^ 2));

mAA = mA.' * mA;
vAb = mA.' * vB;

mObjVal = zeros([numIterations, 2]);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numCols)
    minimize( 0.5 * sum_square( mA * vX - vB ) );
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


%% Solution by Projected Gradient Descent (Alternating Projections)

vX = pinv(mA) * vB;

for ii = 2:numIterations
    
    stepSize = stepSizeBase / sqrt(ii - 1);
    
    % Gradient Step
    vX = vX - (stepSize * ((mAA * vX) - vAb));
    
    % Projection onto Non Negative Orthant
    vX = max(vX, 0);
    % Projection onto Sum of 1
    vX = vX - ((sum(vX) - 1) / numCols);
    
    % Projection onto Non Negative Orthant
    vX = max(vX, 0);
    % Projection onto Sum of 1
    vX = vX - ((sum(vX) - 1) / numCols);
    
    % Projection onto Non Negative Orthant
    vX = max(vX, 0);
    % Projection onto Sum of 1
    vX = vX - ((sum(vX) - 1) / numCols);
    
    mObjVal(ii, 1) = hObjFun(vX);
end

disp([' ']);
disp(['Projected Gradient Descent (Alternating Projection) Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(mObjVal(numIterations, 1))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Projected Gradient Descent (Direct Projection onto Unit Simplex)

vX = pinv(mA) * vB;

for ii = 2:numIterations
    
    stepSize = stepSizeBase / sqrt(ii - 1);
    
    % Gradient Step
    vX = vX - (stepSize * ((mAA * vX) - vAb));
    
    % Projection onto Unit Simplex
    vX = ProjectSimplex(vX, simplexRadius, stopThr);
    
    mObjVal(ii, 2) = hObjFun(vX);
end

disp([' ']);
disp(['Projected Gradient Descent (Direct Projection) Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(mObjVal(numIterations, 2))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Display Results

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot(1:numIterations, [mObjVal, cvx_optval * ones([numIterations, 1])]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(3), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', ['Objective Function Value vs. Iteration'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
set(hAxes, 'XLim', [1, numIterations]);
hLegend = ClickableLegend({['Alternating Projection'], ['Direct Projection'], ['Optimal Value (CVX)']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

