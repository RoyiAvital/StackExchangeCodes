% Mathematics Q454519
% https://math.stackexchange.com/questions/454519
% Least Squares with L2 Linear Norm Regularization (Not Squared)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     22/08/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numRows     = 4;
paramBeta   = 0.5;

numIterations   = 25;


%% Generate Data

vU = randn([numRows, 1]);
vX = 1 + rand([numRows, 1]);

mX = diag(1 ./ vX);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vW(numRows)
    minimize( (0.5 * sum_square(vW - vU)) + (paramBeta * norm(vW ./ vX)) )
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vW.'), ' ]']);
disp([' ']);


%% Solution by Iterative Reweighted Least Squares (IRLS)

hObjFun = @(vW) (0.5 * sum((vW - vU) .^ 2)) + (paramBeta * norm(vW ./ vX));

vObjVal = zeros([numIterations, 1]);

vW = vU;
vObjVal(1) = hObjFun(vW);

for ii = 2:numIterations
    
    % Calculation of the inverse term. Since all diagonlas terms can be
    % done using vectors.
    fctrTerm    = paramBeta / norm(vW ./ vX);
    vXX         = (1 ./ (vX .^ 2));
    vInvXX      = 1 ./ ((fctrTerm * vXX) + ones([numRows, 1]));
    
    vW = vInvXX .* vU;
    
    vObjVal(ii) = hObjFun(vW);
end

disp([' ']);
disp(['Iterative Reweighted Least Squares (IRLS) Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjVal(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vW.'), ' ]']);
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
hLegend = ClickableLegend({['IRLS'], ['Optimal Value (CVX)']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

