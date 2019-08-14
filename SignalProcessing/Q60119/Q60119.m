% StackExchange Signal Processing Q60119
% https://dsp.stackexchange.com/questions/60119
% Estimation / Reconstruction of an Image from Its Missing Data 2D DFT
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     14/08/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

imageName           = 'Lena.png';
rowNum              = 130;
numMissingSamples   = 3;

paramLambda = 0.002;

numIterations   = 1e4;
stepSize        = 2e-5;


%% Generate Data

mI      = mean(im2double(imread(imageName)), 3);
numRows = size(mI, 1);
numCols = size(mI, 2);

rowNum = max(min(rowNum - 1, numRows), 2);

vXRef = mI(rowNum, :);
vXRef = vXRef(:);

mII = repmat(mI, [1, 1, 3]);
mII(rowNum - 1, :, 1) = 1;
mII(rowNum + 1, :, 1) = 1;

figureIdx = figureIdx + 1;

hFigure         = figure();
set(hFigure, 'Units', 'pixels', 'Position', [100, 100, 526, 560]);
hAxes           = axes('Units', 'pixels', 'Position', [9, 9, numCols, numRows]);
hImageObject    = image(mII);
set(get(hAxes, 'Title'), 'String', 'Input Data', 'FontSize', fontSizeTitle);
set(hAxes, 'XTick', []);
set(hAxes, 'XTickLabel', []);
set(hAxes, 'YTick', []);
set(hAxes, 'YTickLabel', []);

if(generateFigures == ON)
    sA = getframe(hFigure);
    imwrite(sA.cdata, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end

vMissingSampledIdx      = randperm(numCols, numMissingSamples);
vY                      = fft(vXRef);
vYRef                   = vY;
vY(vMissingSampledIdx)  = [];

mF = (1 / sqrt(numCols)) * dftmtx(numCols);

vYRef = mF * vXRef;
vY = vYRef;

vY(vMissingSampledIdx)  = [];
mF(vMissingSampledIdx, :) = [];

vD = [1; -1];
mD = spdiags([-ones(numCols, 1), ones(numCols, 1)], [0, 1], numCols - 1, numCols);



%% Reconstruction

vX = abs(mF' * vY);

for ii = 1:numIterations
    vG = abs(mF' * ((mF * vX) - vY)) + (paramLambda * conv2(sign(conv2(vX, vD, 'valid')), flip(vD, 1), 'full'));
    % vG = abs(mF' * ((mF * vX) - vY)) + (paramLambda * mD.' * sign(mD * vX));
    vX = vX - (stepSize * vG);
end

vE = vX - vXRef;

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosMedium);
hAxes   = axes();
% set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(1:numCols, [vXRef, vX]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2), 'LineStyle', ':');
% set(hLineSeries(2), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Estimation of the Data - $ x $ (Row of the Image)']}, ...
    'FontSize', fontSizeTitle, 'Interpreter', 'latex');
set(get(hAxes, 'XLabel'), 'String', {['Pixel Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Pixel Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['Estimation']}, 'Interpreter', 'latex');

if(generateFigures == ON)
    sA = getframe(hFigure);
    imwrite(sA.cdata, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


sqrt(mean(vE .* vE)) * 255
max(abs(vE)) * 255


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

