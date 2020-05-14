% StackExchange Mathematics Q3674241
% https://math.stackexchange.com/questions/3674241
% Solve argmina∥ax−y∥1 - Minimizer of the L1 Norm of the Difference of a Vectors
% References:
%   1.  
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     14/05/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 42; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

numElements = 5; %<! Symmetric Matrix
numPts      = numElements * 1000;


%% Load / Generate Data

vX = randn(numElements, 1);
vY = randn(numElements, 1);

hObjFun = @(valA) sum(abs(valA * vX - vY)); %<! Scaled L1 Norm
hS = @(valA) sign(valA * vX - vY); %<! The 'vS' vector


%% Solution by CVX

solverString = 'CVX';

cvx_solver('SDPT3'); %<! Default, Keep numRows low
% cvx_solver('SeDuMi');
% cvx_solver('Mosek'); %<! Can handle numRows > 500, Very Good!
% cvx_solver('Gurobi');

hRunTime = tic();

cvx_begin('quiet')
% cvx_begin()
    % cvx_precision('best');
    variable valA(1, 1);
    minimize( norm(valA * vX - vY, 1) );
cvx_end

runTime = toc(hRunTime);

% vX = mX(:);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(valA))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(valA), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Solution by Minimum Search

solverString = 'Solution by 1D Search of the Minimum';

hRunTime = tic();

valA = fminsearch(hObjFun, mean(vY ./ vX));

runTime = toc(hRunTime);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(valA))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(valA), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Solution by Root Search

solverString = 'Solution by 1D Search of the Roto of the Gradient';

hF = @(valA) sign(valA * vX - vY).' * vX;

hRunTime = tic();

valA = fzero(hF, mean(vY ./ vX));

runTime = toc(hRunTime);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(valA))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(valA), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Solution by Analytic Form

solverString = 'Solution by Analytic Form';

hRunTime = tic();

valA = SolveScaledL1(vX, vY);

runTime = toc(hRunTime);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(valA))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(valA), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Display Results

vA = sort(vY ./ vX, 'ascend');
vB = zeros(numElements, 1);
vC = zeros(numElements, 1);

for ii = 1:numElements
    vB(ii) = hS(vA(ii)).' * vX;
    vC(ii) = hObjFun(vA(ii));
end

vD = linspace(vA(1) - 1, vA(numElements) + 1, numPts);
vD = sort([vD(:); vA], 'ascend');
vE = zeros(numPts + numElements, 1);
vF = zeros(numPts + numElements, 1);

for ii = 1:(numPts + numElements)
    vE(ii) = hS(vD(ii)).' * vX;
    vF(ii) = hObjFun(vD(ii));
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);

hAxes       = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vD, vE);
set(hLineSeries, 'LineWidth', lineWidthNormal);
hLineSeries = plot(vD, zeros(numPts + numElements, 1));
set(hLineSeries, 'LineWidth', lineWidthNormal);
hLineSeries = plot(vA, vB);
set(hLineSeries, 'LineStyle', 'none', 'Marker', '*', 'MarkerSize', markerSizeLarge);
set(get(hAxes, 'Title'), 'String', ['Value of f''(a)'], 'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'a', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', '$ {f}^{''} \left( a \right) $', ...
    'FontSize', fontSizeAxis, 'Interpreter', 'latex');
hLegend = ClickableLegend({['Values of All Points'], ['Zero Range'], ['Junction Points']});

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);

hAxes       = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(vD, vF);
set(hLineSeries, 'LineWidth', lineWidthNormal);
hLineSeries = plot(vA, vC);
set(hLineSeries, 'LineStyle', 'none', 'Marker', '*', 'MarkerSize', markerSizeLarge);
set(get(hAxes, 'Title'), 'String', ['Value of f(a)'], 'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'a', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', '$ f \left( a \right) $', ...
    'FontSize', fontSizeAxis, 'Interpreter', 'latex');
hLegend = ClickableLegend({['Values of All Points'], ['Junction Points']});

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

