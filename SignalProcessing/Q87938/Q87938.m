% StackExchange Signal Processing Q87938
% https://dsp.stackexchange.com/questions/87938
% Consistent Reconstruction of Image from Partial Images
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     14/07/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants



%% Parameters

% Data
numSamples = 50;
numSignals = 3;

mSeg    = [11, 40; 1, 30; 21, 50];
vDc     = [0.9, 0.8, 0.2];
vAmp    = [0.05, 0.04, 0.06];
vFreq   = [1, 1.1, 0.9];
vPhase  = [0, 0.01, 0.03];
noiseStd = 0.05;
vT = linspace(0, 1, numSamples + 1);
vT = vT(1:(end - 1));
vT = vT(:);

% Model
paramLambda = 400;


%% Generate / Load Data

mX = nan(numSamples, numSignals);

for ii = 1:numSignals
    firstIdx = mSeg(ii, 1);
    lastIdx  = mSeg(ii, 2);
    mX(firstIdx:lastIdx, ii) = vDc(ii) + (vAmp(ii) * sin(2 * pi * vFreq(ii) * vT(firstIdx:lastIdx) + vPhase(ii)));
end


%% Analysis

vX = mean(mX, 2, 'omitnan');
hLossFun = @(vX) ObjFun(vX, mX, paramLambda);

vXX = fminunc(hLossFun, vX);



%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(mX);
for ii = 1:numSignals
    set(hLineObj(ii), 'DisplayName', ['Line ', num2str(ii)]);
end
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = plot(vX, 'DisplayName', 'Mean Line');
set(hLineObj, 'LineWidth', lineWidthNormal, 'LineStyle', ':');

set(get(hAxes, 'Title'), 'String', {['Estimate Data by Partial Data']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);

hLegend = ClickableLegend();

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(mX);
for ii = 1:numSignals
    set(hLineObj(ii), 'DisplayName', ['Line ', num2str(ii)]);
end
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = plot(vXX, 'DisplayName', 'Optimized Line');
set(hLineObj, 'LineWidth', lineWidthNormal, 'LineStyle', ':');

set(get(hAxes, 'Title'), 'String', {['Estimate Data by Partial Data']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);

hLegend = ClickableLegend();

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Auxiliary Functions

function [ objVal ] = ObjFun( vX, mX, paramLambda )

% Fidelity Term
objVal = 0.5 * sum((vX - mX) .^ 2, 'all', 'omitnan');
% Regularization
objVal = objVal + ((paramLambda / 2) * sum((diff(vX) - diff(mX)) .^ 2, 'all', 'omitnan'));

end




%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

