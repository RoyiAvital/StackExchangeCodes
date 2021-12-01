% StackExchange Signal Processing Q79087
% https://dsp.stackexchange.com/questions/79087
% Solving a Weighted Basis Pursuit Denoising Problem (BPDN) with MATLAB / CVX
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     13/11/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numRows = 6;
numCols = 10;

varianceFctr    = 3;
paramLambda     = 2.75;


%% Generate Data

mA  = randn(numRows, numCols);
vX0 = rand(numCols, 1) >= 0.65;
vC  = varianceFctr * rand(numRows, 1);

mCInv = diag(1 ./ vC);

vY = (mA * vX0) + (sqrt(vC) .* randn(numRows, 1));

% Objective Function
hObjFun = @(vX) (0.5 * sum( (vX - vX0) .^ 2 ));


%% Display Data

% figureIdx = figureIdx + 1;
% 
% hFigure = figure('Position', figPosLarge);
% hAxes   = axes(hFigure);
% hLineObj = plot(1:numSamples, [vW, vY]);
% set(hLineObj(1), 'LineWidth', lineWidthNormal);
% set(hLineObj(2), 'LineStyle', 'none', 'Marker', '*');
% set(get(hAxes, 'Title'), 'String', {['Input Signals']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
%     'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['Ground Truth'], ['Input Noisy Samples']});
% 
% if(generateFigures == ON)
%     % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
%     print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
% end


%% Solution by CVX

solverString = 'CVX';

% cvx_solver('SDPT3'); %<! Default, Keep numRows low
% cvx_solver('SeDuMi');
% cvx_solver('Mosek'); %<! Can handle numRows > 500, Very Good!
% cvx_solver('Gurobi');

hRunTime = tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numCols);
    minimize( 0.5 * quad_form(mA * vX - vY, mCInv) + (paramLambda * norm(vX, 1)) );
cvx_end

runTime = toc(hRunTime);

DisplayRunSummary(solverString, hObjFun, vX, runTime, cvx_status);

sCvxSol.vXCvx     = vX(:);
sCvxSol.cvxOptVal = hObjFun(vX);




%% Display Results

% figureIdx = figureIdx + 1;
% 
% hFigure = DisplayComparisonSummary(numIterations, mObjFunValMse, mSolMse, cLegendString, figPosLarge, lineWidthNormal, fontSizeTitle, fontSizeAxis);
% 
% if(generateFigures == ON)
%     % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
%     print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
% end
% 
% 
% figureIdx = figureIdx + 1;
% 
% hFigure = figure('Position', figPosLarge);
% hAxes   = axes(hFigure);
% hLineObj = plot(1:numSamples, [vW, vY, vX]);
% set(hLineObj(1), 'LineWidth', lineWidthNormal);
% set(hLineObj(2), 'LineStyle', 'none', 'Marker', '*');
% % set(hLineObj(3), 'LineWidth', lineWidthThin, 'LineStyle', ':');
% set(hLineObj(3), 'LineStyle', 'none', 'Marker', 'x');
% set(get(hAxes, 'Title'), 'String', {['Signals']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
%     'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['Ground Truth'], ['Input Noisy Samples'], ['TV Estimation']});
% 
% if(generateFigures == ON)
%     % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
%     print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
% end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

