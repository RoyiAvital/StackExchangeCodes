% Mathematics Q2595199
% https://math.stackexchange.com/questions/2595199
% Proximal Mapping of Least Squares with L1 and L2 Mixed Norm Regularization (Elastic Net)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     13/03/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

paramLam1 = 1;
paramLam2 = 2;

numElements = 4;

numIterations   = 6000;
stepSize        = 0.00095;


%% Generate Data

vB = 10 * randn([numElements, 1]);

hObjFun = @(vX) (0.5 * sum((vX - vB) .^ 2)) + (paramLam1 * norm(vX, 1)) + (paramLam2 * norm(vX, 2));
hL2SubGrad = @(vX) (norm(vX, 2) == 0) * (vX ./ (norm(vX, 2) + ((norm(vX, 2) == 0) * 1e-7)));


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numElements)
    minimize( (0.5 * square_pos(norm(vX - vB, 2))) + (paramLam1 * norm(vX, 1)) + (paramLam2 * norm(vX, 2)) );
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Solution by Sub Gradient Method

vObjValSgm = zeros([numIterations, 1]);

vX = zeros([numElements, 1]);
vObjValSgm(1) = hObjFun(vX);

for ii = 2:numIterations
    vG = (vX - vB) + (paramLam1 * sign(vX)) + (paramLam2 * hL2SubGrad(vX));
    vX = vX - (stepSize * vG);
    
    vObjValSgm(ii) = hObjFun(vX);
    
end

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot([1:numIterations], [cvx_optval * ones([numIterations, 1]), vObjValSgm]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', ['Least Squares with Mixed Norm Regularization'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Index', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['CVX'], ['Sub Gradient Method']});
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


%% Solution by Proximal Gradient Method




%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

