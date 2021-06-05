% StackExchange Signal Processing Q75536
% https://dsp.stackexchange.com/questions/75536
% Locate Non Homogenous Areas in an Image
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     04/06/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

imageFileName = 'Lena256.png';

numSuperPixels  = 200;
filterStd       = 2;
noiseThr        = 5 / 255;
patchSize       = 7;


%% Generate Data

mI = im2double(imread(imageFileName));

[numRows, numCols] = size(mI);

[mL, numLabels] = superpixels(mI, numSuperPixels);

mM  = imgaussfilt(mI, filterStd); %<! Local Mean
mM2 = imgaussfilt(mI .^ 2, filterStd); %<! The 2nd Moment
mV  = max(mM2 - (mM .^ 2), 0); %<! Local Variance


%% Display Data - Variance

figureIdx = figureIdx + 1;

hFigure = figure('Position', [100, 100, 542, 642]);
hAxes   = axes(hFigure, 'Units', 'pixels', 'Position', [010, 326, 256, 256]);
hImgObj = image(repmat(mI, [1, 1, 3]));
set(get(hAxes, 'Title'), 'String', {['Input Image']}, ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTick', [], 'XTickLabel', []);
set(hAxes, 'YTick', [], 'YTickLabel', []);

hAxes   = axes(hFigure, 'Units', 'pixels', 'Position', [276, 326, 256, 256]);
% hImgObj = imagesc(repmat(mV, [1, 1, 3]));
hImgObj = imagesc(mV);
set(get(hAxes, 'Title'), 'String', {['Local Variance']}, ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTick', [], 'XTickLabel', []);
set(hAxes, 'YTick', [], 'YTickLabel', []);

drawnow();


%% Solution by Super Pixels

mSPV = zeros(numRows, numCols); %<! Variance of a Super Pixel

for ii = 1:numLabels
    vSuperPixelIdx          = mL(:) == ii;
    mSPV(vSuperPixelIdx)    = var(mI(vSuperPixelIdx));
end


%% Display Result

hAxes   = axes(hFigure, 'Units', 'pixels', 'Position', [010, 010, 256, 256]);
hImgObj = imagesc(mSPV);
set(get(hAxes, 'Title'), 'String', {['Variance by Super Pixel']}, ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTick', [], 'XTickLabel', []);
set(hAxes, 'YTick', [], 'YTickLabel', []);

drawnow();


%% Solution by Noise Estimation and Weak Texture

mT = WeakTextureMask(mI, noiseThr, patchSize);


%% Display Results

hAxes   = axes(hFigure, 'Units', 'pixels', 'Position', [276, 010, 256, 256]);
hImgObj = imagesc(~mT);
set(get(hAxes, 'Title'), 'String', {['Non Weak Texture Image']}, ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTick', [], 'XTickLabel', []);
set(hAxes, 'YTick', [], 'YTickLabel', []);

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

