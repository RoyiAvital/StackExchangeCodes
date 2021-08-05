function [ hF ] = DisplayComparisonSummary( numIterations, mObjFunValMse, mSolMse, cLegendString, figPos, lineWidth, fontSizeTitle, fontSizeAxis )
% ----------------------------------------------------------------------------------------------- %
% Remarks:
%   1.  T
% Known Issues:
%   1.  A
% TODO:
%   1.  A
% Release Notes:
%   -   1.1.000     26/12/2020
%       *   Using MSE / Squared Norm.
%   -   1.0.000     23/11/2016
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

hF     = figure('Position', figPos);

hAxes       = subplot(2, 1, 1);
hLineSeries = plot(1:numIterations, 10 * log10(mObjFunValMse));
set(hLineSeries, 'LineWidth', lineWidth);
set(get(hAxes, 'Title'), 'String', ['Objective Function Value vs. Optimal Value (CVX)'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', '$ 10 \log_{10} \left( {\left| f \left( x \right) - f \left( {x}_{CVX} \right) \right|}^{2} \right) $', ...
    'FontSize', fontSizeAxis, 'Interpreter', 'latex');
set(hAxes, 'XLim', [1, numIterations]);
hLegend = ClickableLegend(cLegendString);

hAxes       = subplot(2, 1, 2);
hLineSeries = plot(1:numIterations, 10 * log10(mSolMse));
set(hLineSeries, 'LineWidth', lineWidth);
set(get(hAxes, 'Title'), 'String', ['Solution Error Norm'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', '$ 10 \log_{10} \left( {\left\| x - {x}_{CVX} \right\|}_{2}^{2} \right) $', ...
    'FontSize', fontSizeAxis, 'Interpreter', 'latex');
set(hAxes, 'XLim', [1, numIterations]);
hLegend = ClickableLegend(cLegendString);


end

