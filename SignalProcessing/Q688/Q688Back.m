% StackExchange Signal Processing Q688
% https://dsp.stackexchange.com/questions/688
% What Is the Algorithm Behind Photoshop's �Black and White� Adjustment Layer?
% References:
%   1.  A
% Remarks:
%   1.  HSL Picker - http://hslpicker.com/.
%   2.  Colorizer Color Picker - http://colorizer.org/.
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     22/12/2018
%   *   First release.


%% General Parameters

subStreamNumberDefault = 2123;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;
generateImages  = OFF;


%% Simulation Parameters

vPhotoshopValues = randi([0, 100], [6, 1]);
% vPhotoshopValues = [56; 61; 50; 78; 76; 54];
vPhotoshopValues = [30; 98; 51; 73; 5; 53]; %<! This is used in Photoshop

% 11 Colors - [H, 1, 0.5] Where H goes in 30% Steps -> [0, 30, 60, ..., 330]
% Each color is a row.
mBaseColors = [1, 0, 0; 1, 0.5, 0; 1, 1, 0; 0.5, 1, 0; 0, 1, 0; 0, 1, 0.5; 0, 1, 1; 0, 0.5, 1; 0, 0, 1; 0.5, 0, 1; 1, 0, 1; 1, 0, 0.5];

mBaseColors = rand(21, 3);


cellSize    = 50; %<! Pixels


%% Generate Data

numColors = size(mBaseColors, 1);

numRows = cellSize;
numCols = cellSize * numColors;

mI = zeros(numRows, numCols, 3);

for ii = 1:numColors
    vCurrColor = mBaseColors(ii, :);
    firstColIdx = ((ii - 1) * cellSize) + 1;
    lastColIdx  = ii * cellSize;
    mI(:, firstColIdx:lastColIdx, :) = repmat(reshape(vCurrColor, [1, 1, 3]), [cellSize, cellSize 1]);
end

% vCoeffValues = [0.5; 0; 0; 0; 0; 0];
vCoeffValues = (vPhotoshopValues - 50) ./ 50;
mO = ApplyBlackWhiteFilter(mI, vCoeffValues);

% figure;
% imshow(mI);
% figure;
% imshow(mO);
% 
% figure;
% imshow(im2uint8(mO));

if(generateImages == ON)
    imwrite(im2uint8(mI), 'ReferenceImage2.png');
end


%% Analysis vs. Photoshop

mORef   = im2single(imread('PhotoshopImage2.png'));
mE      = mORef - mO;

maxAbsDev = max(abs(mE(:)));


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = subplot(4, 1, 1);
hImgObj = imshow(mI);
set(get(hAxes, 'Title'), 'String', {['Reference Image']}, ...
    'FontSize', fontSizeTitle);

hAxes   = subplot(4, 1, 2);
hImgObj = imshow(mO);
set(get(hAxes, 'Title'), 'String', {['MATLAB Output'], ['Black & White Adjustment Layer Input Values - ', num2str(vPhotoshopValues.')]}, ...
    'FontSize', fontSizeTitle);

hAxes   = subplot(4, 1, 3);
hImgObj = imshow(mORef);
set(get(hAxes, 'Title'), 'String', {['Photoshop Output'], ['Black & White Adjustment Layer Input Values - ', num2str(vPhotoshopValues.')]}, ...
    'FontSize', fontSizeTitle);

hAxes   = subplot(4, 1, 4);
hImgObj = imshow(mE);
set(get(hAxes, 'Title'), 'String', {['Error Image (Absolute Deviation)'], ['Max Absolute Deviation - ', num2str(maxAbsDev)]}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

