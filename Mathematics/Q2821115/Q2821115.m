% Mathematics Q2821115
% https://math.stackexchange.com/questions/2821115
% Maximum Likelihood Estimator (MLE) of \theta for the PDF f(x; \theta) = 0.5 (1 + \theta x)
% References:
%   1.  aa
% Remarks:
%   1.  See
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     16/06/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Symbolic Analysis

syms varU varS varX paramTheta

funPdf(varS, paramTheta) = 0.5 * (1 + paramTheta * varS);
funCdf = int(funPdf, varS, -1, varX);
funTranformation = root(funCdf - varU, varX);

% The Solution to the Transformation (Both are equivalent)
% varX = (sqrt(paramTheta * paramTheta + paramTheta * (4 * valU - 2) + 1) - 1) / paramTheta;
% varX = (sqrt(4 * paramTheta * varU - 2 * paramTheta + paramTheta ^ 2 + 1)
% - 1) / paramTheta;

% Verification by:
% http://www.wolframalpha.com/input/?i=integrate+0.5+*+(1+%2B+%5Ctheta+*+s)+ds+from+-1+to+x
% http://www.wolframalpha.com/input/?i=solve+0.5+%2B+0.5+x+-+0.25+%CE%B8+%2B+0.25+x%5E2+%CE%B8+-+U+for+x


%% Simulation Parameters

numSamples = 1e6;
paramTheta = 0.3;

gridSamplesData     = 100;
gridSamplesTheta    = 200;


%% Generate Data

vDataGrid   = linspace(-1, 1, gridSamplesData);
vDataGrid   = vDataGrid(:);
vThetaGrid  = linspace(-1, 1, gridSamplesTheta);
vThetaGrid  = vThetaGrid(:);

vU = rand([numSamples, 1]);
vX = (sqrt(paramTheta * paramTheta + paramTheta * (4 * vU - 2) + 1) - 1) / paramTheta;

[vBinFreq, vBinCenter] = hist(vX, gridSamplesData);
vBinFreq = vBinFreq ./ numSamples;

delatX = mean(diff(vDataGrid));

hObjFun = @(paramTheta) sum(vX ./ (1 + paramTheta * vX));
hMlFun = @(paramTheta) sum(log(1 + paramTheta * vX));

vMlFunVal = zeros([gridSamplesTheta, 1]);
for ii = 1:gridSamplesTheta
    vMlFunVal(ii) = hMlFun(vThetaGrid(ii));
end

paramThetaEst = fzero(hObjFun, 0);


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes = subplot(2, 1, 1);
set(hAxes, 'NextPlot', 'add');
hBarObj = bar(vBinCenter, vBinFreq);
hLineObj = line(vDataGrid, 0.5 * (1 + paramTheta * vDataGrid) * delatX);
set(hLineObj, 'LineWidth', lineWidthNormal, 'Color', 'r');
set(get(hAxes, 'Title'), 'String', {['Histogram of the PDF - $f \left( x \right) = \frac{1}{2} \left( 1 + \theta x \right)$']}, ...
    'FontSize', fontSizeTitle, 'Interpreter', 'latex');
set(get(hAxes, 'XLabel'), 'String', 'Value', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Probability', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Empirical Histogram'], ['Analytic PDF - \theta = ', num2str(paramTheta)]});

hAxes = subplot(2, 1, 2);
set(hAxes, 'NextPlot', 'add');
hLineObj = line(vThetaGrid, vMlFunVal);
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = line(paramTheta, hMlFun(paramTheta));
set(hLineObj, 'LineStyle', 'none', 'Marker', '*', 'MarkerSize', markerSizeNormal, 'LineWidth', 4, 'Color', mColorOrder(2, :));
hLineObj = line(paramThetaEst, hMlFun(paramThetaEst));
set(hLineObj, 'LineStyle', 'none', 'Marker', 'o', 'MarkerSize', markerSizeNormal, 'LineWidth', 4, 'Color', mColorOrder(3, :));
set(get(hAxes, 'Title'), 'String', {['Maximum Likelihood Function']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', '\theta', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Log Likelihood', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Log Likelihood'], ['Ground Truth'], ['Estimation']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

