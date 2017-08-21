% Mathematics Q2399321
% https://math.stackexchange.com/questions/2399321
% Least Squares with Euclidean L2 Norm Constraint
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     21/08/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numRows = 4;
numCols = 3; %<! Number of Vectors - i (K in the question)


%% Generate Data

mA = randn([numRows, numCols]);
vB = randn([numRows, 1]);

normConst = 1;


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numCols)
    minimize( square_pos(  norm(mA * vX - vB, 2) ) );
    subject to
        norm(vX, 2) <= normConst;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by KKT Solution

vParamLambda    = [0:0.1:25];
vNormVal        = zeros([length(vParamLambda), 1]);

mAA = mA.' * mA;
mAb = mA.' * vB;
mI  = eye(size(mA, 2));

for ii = 1:length(vParamLambda)
    paramLambda = vParamLambda(ii);
    
    vNormVal(ii) = norm((mAA + (paramLambda * mI)) \ mAb, 2);
    
end

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot(vParamLambda, [vNormVal, normConst * ones([length(vParamLambda), 1])]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', ['Tikhonov Regularization Least Squares Solution Norm'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', '\lambda', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'L_2 Norm', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Solution Norm'], ['Constraint Norm']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end



vX = SolveLsNormConst(mA, vB, normConst);

objVal = sum((mA * vX - vB) .^ 2);

disp([' ']);
disp(['Projected Gradient Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(objVal)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

