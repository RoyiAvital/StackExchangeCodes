% StackExchange Signal Processing Q63449
% https://dsp.stackexchange.com/questions/63449
% Deconvolution of an Image Acquired by a Square Uniform Detector
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     25/01/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Simulation Parameters

imageFileName   = 'Lenna256.png';
kernelRadius    = 1;
convShape       = CONVOLUTION_SHAPE_VALID;

paramLambda = 0.005;

maxSize = 64;


%% Generate Data

mA = im2double(imread(imageFileName));
numRows = size(mA, 1);
numCols = size(mA, 2);

imgSize = min(numRows, numCols);
mA = mA(1:imgSize, 1:imgSize, :);
mA = imresize(mA, [maxSize, maxSize]);

mK = ones((2 * kernelRadius) + 1);
% mK = rand((2 * kernelRadius) + 1);
mK = mK / sum(mK(:));

switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        convShapeString = 'full';
    case(CONVOLUTION_SHAPE_SAME)
        convShapeString = 'same';
    case(CONVOLUTION_SHAPE_VALID)
        convShapeString = 'valid';
end

hFigure     = figure();
hAxes       = axes();
hImageObj   = imshow(mA);
set(get(hAxes, 'Title'), 'String', {['Input Image - Lenna']}, ...
    'FontSize', fontSizeTitle);

mB = conv2(mA, mK, convShapeString);

hFigure     = figure();
hAxes       = axes();
hImageObj   = imshow(mB);
set(get(hAxes, 'Title'), 'String', {['Sensor Image - Lenna']}, ...
    'FontSize', fontSizeTitle);


%% Solution by Linear Algebra

mKK = CreateConvMtx2D(mK, maxSize, maxSize, convShape);
% Basically:
% vB = mKK * mA(:);
vA = mKK \ mB(:);
% vA = pinv(full(mKK)) * mB(:);

vA = ((mKK.' * mKK) + (paramLambda * speye(maxSize * maxSize))) \ (mKK.' * mB(:));

mAA = reshape(vA, maxSize, maxSize); %<! Restored Image

hFigure     = figure();
hAxes       = axes();
hImageObj   = imshow(mAA);
set(get(hAxes, 'Title'), 'String', {['Estimated Image - Lenna']}, ...
    'FontSize', fontSizeTitle);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

