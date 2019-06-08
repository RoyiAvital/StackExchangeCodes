% StackExchange Signal Processing Q58730
% https://dsp.stackexchange.com/questions/58730
% How Much Zero Padding Do We Need to Perform Filtering in the Fourier Domain?
% References:
%   1.  See Applying Low Pass and Laplace of Gaussian Filter in Frequency Domain - https://stackoverflow.com/questions/50614085.
%       Under my solution, the script 'FreqDomainConv.m'.
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     04/04/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

PADDING_MODE_ZEROS      = 1;
PADDING_MODE_SYMMETRIC  = 2;
PADDING_MODE_REPLICATE  = 3;
PADDING_MODE_CIRCULAR   = 4;


%% Simulation Parameters

imageFolder             = 'ImageSet';
imageFileNamePattern    = '*.jpg';
numImgsRowDisplay       = 3;
numImgsColDisplay       = 3;
numImgDisplay           = numImgsRowDisplay * numImgsColDisplay;

vNumComponenets = [1, 5, 10, 25, 50, 75, 100, 1e100];

refImg = 1;


%% Load Data

sImageFile = dir(fullfile(imageFolder, imageFileNamePattern));
numImages = length(sImageFile);

mI          = im2double(imread(fullfile(imageFolder, sImageFile(1).name)));
numRows     = size(mI, 1);
numCols     = size(mI, 2);
numChannels = size(mI, 3);
numPixels   = numRows * numCols;
numElements = numPixels * numChannels;

vNumComponenets = min(vNumComponenets, numImages);

hVecToImg = @(vI) reshape(vI, numRows, numCols, numChannels);
hVecChannelToImg = @(vR, vG, vB) hVecToImg([vR; vG; vB]);


%% Image Vectorization

mI = zeros(numElements, numImages, 'single');

for ii = 1:numImages
    mI(:, ii) = reshape(im2single(imread(fullfile(imageFolder, sImageFile(ii).name))), numElements, 1);
end

% Centering
vMeanI = mean(mI, 2);
mI = mI - vMeanI;

% Without the 'econ' flag the SVD requires 17 GB of Memory. If you don't
% have 32 GB machine to the least, don't run it!
[mU, mS, mV] = svd(mI, 'econ');
mC = mS * mV.';

% Display Results
figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosX2Large);
imgIdx = 0;

for ii = 1:numImgsRowDisplay
    for jj = 1:numImgsColDisplay
        imgIdx  = imgIdx + 1;
        hAxes   = subplot(3, 3, imgIdx);
        if(imgIdx == 1)
            hImgObj = image(hVecToImg(mI(:, refImg) + vMeanI));
            set(get(hAxes, 'Title'), 'String', {['Original Image']}, ...
                'FontSize', fontSizeTitle);
            set(hAxes, 'DataAspectRatio', [1, 1, 1]);
        else
            numComponents = vNumComponenets(imgIdx - 1);
            vComIdx = [1:numComponents];
            hImgObj = image(hVecToImg((mU(:, vComIdx) * mC(vComIdx, refImg)) + vMeanI));
            set(get(hAxes, 'Title'), 'String', {['Number of Components - ', num2str(numComponents)]}, ...
                'FontSize', fontSizeTitle);
            set(hAxes, 'DataAspectRatio', [1, 1, 1]);
        end
        set(hAxes, 'XTick', []);
        set(hAxes, 'XTickLabel', []);
        set(hAxes, 'YTick', []);
        set(hAxes, 'YTickLabel', []);
    end
end

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end

% Working Per Channel

% Red Channel
mIR = mI(1:numPixels, :);
vMeanIR = vMeanI(1:numPixels);
[mUR, mSR, mVR] = svd(mIR, 'econ');
mCR = mSR * mVR.';

% Green Channel
mIG = mI(numPixels + 1:2 * numPixels, :);
vMeanIG = vMeanI(numPixels + 1:2 * numPixels);
[mUG, mSG, mVG] = svd(mIG, 'econ');
mCG = mSG * mVG.';

% Blue Channel
mIB = mI(2 * numPixels + 1:3 * numPixels, :);
vMeanIB = vMeanI(2 * numPixels + 1:3 * numPixels);
[mUB, mSB, mVB] = svd(mIB, 'econ');
mCB = mSB * mVB.';

% Display Results
figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosX2Large);
imgIdx = 0;

for ii = 1:numImgsRowDisplay
    for jj = 1:numImgsColDisplay
        imgIdx  = imgIdx + 1;
        hAxes   = subplot(3, 3, imgIdx);
        if(imgIdx == 1)
            hImgObj = image(hVecToImg(mI(:, refImg) + vMeanI));
            set(get(hAxes, 'Title'), 'String', {['Original Image']}, ...
                'FontSize', fontSizeTitle);
            set(hAxes, 'DataAspectRatio', [1, 1, 1]);
        else
            numComponents = vNumComponenets(imgIdx - 1);
            vComIdx = [1:numComponents];
            hImgObj = image(hVecChannelToImg((mUR(:, vComIdx) * mCR(vComIdx, refImg)) + vMeanIR, ...
                (mUG(:, vComIdx) * mCG(vComIdx, refImg)) + vMeanIG, ...
                (mUB(:, vComIdx) * mCB(vComIdx, refImg)) + vMeanIB));
            set(get(hAxes, 'Title'), 'String', {['Number of Components - ', num2str(numComponents)]}, ...
                'FontSize', fontSizeTitle);
            set(hAxes, 'DataAspectRatio', [1, 1, 1]);
        end
        set(hAxes, 'XTick', []);
        set(hAxes, 'XTickLabel', []);
        set(hAxes, 'YTick', []);
        set(hAxes, 'YTickLabel', []);
    end
end

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

