% Test Run Time of Various Ridge Regression Solvers
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     18/07/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

vN          = [500; 1000; 2000; 2500; 3000; 3500; 4000; 5000; 7500; 10000];
vLambda     = linspace(0, 2, 20).';
randFctr    = 1e-5;

cSolvers = {@(mA, vB, paramLambda) SolveRidgeReg(mA, vB, paramLambda); ...
            @(mA, vB, paramLambda) lsmb(mA, vB, paramLambda * paramLambda); ...
            @(mA, vB, paramLambda) lsmr(mA, vB, paramLambda * paramLambda)};


%% Generate Data

mT = zeros(size(vLambda, 1), size(vN, 1), size(cSolvers, 1));


%% Run Time Analysis

for kk = 1:size(cSolvers, 1)
    for jj = 1:size(vN, 1)
        mA = sprand(vN(jj), vN(jj), randFctr);
        vB = randn(vN(jj), 1);
        for ii = 1:size(vLambda, 1)
            paramLambda = vLambda(ii);
            hF = @() cSolvers{kk}(mA, vB, paramLambda);
            mT(ii, jj, kk) = TimeItMin(hF);
        end
    end
end


%% Display Results

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100, 100, 800, 600]);
hAxes       = axes(hFigure);
hLineObj    = plot(1:numSamples, [vX, vY]);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Data']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth Signal'], ['Measured Signal']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

figureIdx = figureIdx + 1;

vH(numSamples) = 0;

hFigure     = figure('Position', [100, 100, 800, 600]);
hAxes       = axes(hFigure);
PlotDft([vX, vY, vH], 1, 'normalizDataFlag', 1, 'removeDc', 1, 'plotTitle', 'The DFT of the Data and Filter', 'plotLegendFlag', 1,'plotLegend', {['GT Signal'], ['Measured Signal'], ['Filter']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100, 100, 800, 600]);
hAxes       = axes(hFigure);
hLineObj    = plot(1:numSamples, [vX, vY, vXEstWoReg, vXEstWReg]);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Comparison of Deconvolution Methods']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['Input Signal'], ['Estimated without Regularization'], ['Estimated with Regularization']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

