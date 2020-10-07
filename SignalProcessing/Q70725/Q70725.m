% StackExchange Signal Processing Q70725
% https://dsp.stackexchange.com/questions/70725
% The Effect of the Standard Deviation (σ) of a Gaussian Kernel when Smoothing a Gradients Image
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     07/10/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;

STD_TO_RADIUS_FACTOR = 5;


%% Simulation Parameters

vStdVal = [1:5];


%% Generate Data

numStd          = length(vStdVal);
maxStd          = max(vStdVal);
kernelRadius    = ceil(STD_TO_RADIUS_FACTOR * maxStd);
kernelLength    = (2 * kernelRadius) + 1;
vX              = [-kernelRadius:kernelRadius].';

mK = zeros(kernelLength, numStd);

for ii = 1:numStd
    mK(:, ii) = exp(-(vX .^ 2) / (2 * vStdVal(ii) * vStdVal(ii)));
end

mK = mK ./ sum(mK, 1); %<! Normalize to sum of 1

cLgenedText = cell(numStd, 1);

for ii = 1:numStd
    cLgenedText{ii} = ['\sigma = ', num2str(vStdVal(ii))];
end


%% Display Kernels

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosDefault); %<! [x, y, width, height]
hAxes       = axes(); %<! [x, y, width, height]
hLineObj    = plot(vX, mK);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Gaussian Kernel as a Function of \sigma']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Support Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Kernel Value']}, ...
    'FontSize', fontSizeAxis);
% set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
hLegend = ClickableLegend(cLgenedText);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Step Function

vY = zeros(101, 1);
vY(45:55) = 1;

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosDefault); %<! [x, y, width, height]
hAxes       = axes(); %<! [x, y, width, height]
hLineObj    = plot(vY);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Step Function']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
% set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
% hLegend = ClickableLegend({['\sigma = ', num2str(vStdVal(1))], ['\sigma = ', num2str(vStdVal(2))], ['\sigma = ', num2str(vStdVal(3))]});

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Convolution

mZ = zeros(length(vY) - size(mK, 1) + 1, numStd);

for ii = 1:numStd
    mZ(:, ii) = conv(vY, mK(:, ii), 'valid');
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100   100   760   480]); %<! [x, y, width, height]
hAxes       = axes(); %<! [x, y, width, height]
hLineObj    = plot(mZ);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Convolution of a Step Function with Gaussian Kernel as a Function of \sigma']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
% set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
hLegend = ClickableLegend(cLgenedText);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end



%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

