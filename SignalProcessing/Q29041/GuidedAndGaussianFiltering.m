% Comparing Gaussian Blur and Guided Image Filtering
% See http://dsp.stackexchange.com/questions/29041

%% General Parameters and Initialization

clear();
close('all');

% set(0, 'DefaultFigureWindowStyle', 'docked');
defaultLooseInset = get(0, 'DefaultAxesLooseInset');
% set(0, 'DefaultAxesLooseInset', [0.05, 0.05, 0.05, 0.05]);

titleFontSize   = 14;
axisFotnSize    = 12;
stringFontSize  = 12;

thinLineWidth   = 2;
normalLineWidth = 3;
thickLineWidth  = 4;

smallSizeData   = 36;
mediumSizeData  = 48;
bigSizeData     = 60;

randomNumberStream  = RandStream('mlfg6331_64', 'NormalTransform', 'Ziggurat');
subStreamNumber     = round(sum(clock()));
% subStreamNumber    = 57162;
% subStreamNumber    = 2143;
set(randomNumberStream, 'Substream', subStreamNumber);
RandStream.setGlobalStream(randomNumberStream);


%% Setting Constants

FALSE   = 0;
TRUE    = 1;

OFF = 0;
ON  = 1;

RADIUS_TO_STD_FACTOR    = 5;


%% Setting Parameters

numRows = 400;
numCols = 400;

grayLevel1 = 245 / 255;
grayLevel2 = 10 / 255;

noiseStd = 10 / 255;

refLine1Row     = 201;
vRefLine1ColIdx = [1:numCols];

gaussianFilterRadius    = 7;
vGuidedFilterRadius     = [7, 7];
guidedFilterSmoothing   = 0.01;


%% Creating Data

borderColIdx = round(numCols / 2);

% Refrence Image
mRefImage = zeros([numRows, numCols]);
mRefImage(:, 1:borderColIdx)            = grayLevel1;
mRefImage(:, (borderColIdx + 1):end)    = grayLevel2;

% Noisy Image
mNoisyImage = mRefImage + (noiseStd * randn([numRows, numCols]));

% Gaussian Filtered Image
mGaussFilteredImage = ApplyGaussianBlur(mNoisyImage, gaussianFilterRadius, RADIUS_TO_STD_FACTOR);

% Guided Filtered Image
mGuidedFilteredImage = imguidedfilter(mNoisyImage, 'NeighborhoodSize', vGuidedFilterRadius, 'DegreeOfSmoothing', guidedFilterSmoothing);


%% Displaying Results

% Displaying Reference Image
mImgDisplay                                     = repmat(mRefImage, [1, 1, 3]);
mImgDisplay(refLine1Row, vRefLine1ColIdx, 1)    = 1;
mImgDisplay(refLine1Row, vRefLine1ColIdx, 2)    = 0;
mImgDisplay(refLine1Row, vRefLine1ColIdx, 3)    = 0;
for iRow = 1:numRows
    mImgDisplay(iRow, iRow, 1) = 0;
    mImgDisplay(iRow, iRow, 2) = 1;
    mImgDisplay(iRow, iRow, 3) = 0;
end

hFigure         = figure();
set(hFigure, 'Units', 'pixels', 'Position', [100, 100, 500, 500]);
hAxes           = axes('Units', 'pixels', 'Position', [50, 50, numCols, numRows]);
hImageObject    = image(mImgDisplay);
set(get(hAxes, 'Title'), 'String', 'Reference Image', 'FontSize', titleFontSize);

% Extract Line 01
vLine1 = mRefImage(refLine1Row, vRefLine1ColIdx);
% Extract Line 02
vLine2 = zeros([numRows, 1]);
for iRow = 1:numRows
    vLine2(iRow) = mRefImage(iRow, iRow);
end

hFigure = figure();
hAxes   = axes();
hLineSeries = plot([1:numCols], vLine1, [1:numRows], vLine2);
set(hLineSeries(1), 'LineWidth', normalLineWidth, 'Color', 'r');
set(hLineSeries(2), 'LineWidth', normalLineWidth, 'Color', 'g');
set(get(hAxes, 'Title'), 'String', 'Values Across Lines - Reference Image', 'FontSize', titleFontSize);
hLegend = legend({['Line #01'], ['Line #02']});
set(hLegend, 'FontSize', axisFotnSize);


% Displaying Noisy Image
mImgDisplay                                     = repmat(mNoisyImage, [1, 1, 3]);
mImgDisplay(refLine1Row, vRefLine1ColIdx, 1)    = 1;
mImgDisplay(refLine1Row, vRefLine1ColIdx, 2)    = 0;
mImgDisplay(refLine1Row, vRefLine1ColIdx, 3)    = 0;
for iRow = 1:numRows
    mImgDisplay(iRow, iRow, 1) = 0;
    mImgDisplay(iRow, iRow, 2) = 1;
    mImgDisplay(iRow, iRow, 3) = 0;
end

hFigure         = figure();
set(hFigure, 'Units', 'pixels', 'Position', [100, 100, 500, 500]);
hAxes           = axes('Units', 'pixels', 'Position', [50, 50, numCols, numRows]);
hImageObject    = image(mImgDisplay);
set(get(hAxes, 'Title'), 'String', 'Noisy Image', 'FontSize', titleFontSize);

% Extract Line 01
vLine1 = mNoisyImage(refLine1Row, vRefLine1ColIdx);
% Extract Line 02
vLine2 = zeros([numRows, 1]);
for iRow = 1:numRows
    vLine2(iRow) = mNoisyImage(iRow, iRow);
end

hFigure = figure();
hAxes   = axes();
hLineSeries = plot([1:numCols], vLine1, [1:numRows], vLine2);
set(hLineSeries(1), 'LineWidth', normalLineWidth, 'Color', 'r');
set(hLineSeries(2), 'LineWidth', normalLineWidth, 'Color', 'g');
set(get(hAxes, 'Title'), 'String', 'Values Across Lines - Noisy Image', 'FontSize', titleFontSize);
hLegend = legend({['Line #01'], ['Line #02']});
set(hLegend, 'FontSize', axisFotnSize);

% Displaying Gaussian Filtered Image
mImgDisplay                                     = repmat(mGaussFilteredImage, [1, 1, 3]);
mImgDisplay(refLine1Row, vRefLine1ColIdx, 1)    = 1;
mImgDisplay(refLine1Row, vRefLine1ColIdx, 2)    = 0;
mImgDisplay(refLine1Row, vRefLine1ColIdx, 3)    = 0;
for iRow = 1:numRows
    mImgDisplay(iRow, iRow, 1) = 0;
    mImgDisplay(iRow, iRow, 2) = 1;
    mImgDisplay(iRow, iRow, 3) = 0;
end

hFigure         = figure();
set(hFigure, 'Units', 'pixels', 'Position', [100, 100, 500, 500]);
hAxes           = axes('Units', 'pixels', 'Position', [50, 50, numCols, numRows]);
hImageObject    = image(mImgDisplay);
set(get(hAxes, 'Title'), 'String', 'Gaussian Filtered Image', 'FontSize', titleFontSize);

% Extract Line 01
vLine1 = mGaussFilteredImage(refLine1Row, vRefLine1ColIdx);
% Extract Line 02
vLine2 = zeros([numRows, 1]);
for iRow = 1:numRows
    vLine2(iRow) = mGaussFilteredImage(iRow, iRow);
end

hFigure = figure();
hAxes   = axes();
hLineSeries = plot([1:numCols], vLine1, [1:numRows], vLine2);
set(hLineSeries(1), 'LineWidth', normalLineWidth, 'Color', 'r');
set(hLineSeries(2), 'LineWidth', normalLineWidth, 'Color', 'g');
set(get(hAxes, 'Title'), 'String', 'Values Across Lines - Gaussian Filtered Image', 'FontSize', titleFontSize);
hLegend = legend({['Line #01'], ['Line #02']});
set(hLegend, 'FontSize', axisFotnSize);

% Displaying Guided Filtered Image
mImgDisplay                                     = repmat(mGuidedFilteredImage, [1, 1, 3]);
mImgDisplay(refLine1Row, vRefLine1ColIdx, 1)    = 1;
mImgDisplay(refLine1Row, vRefLine1ColIdx, 2)    = 0;
mImgDisplay(refLine1Row, vRefLine1ColIdx, 3)    = 0;
for iRow = 1:numRows
    mImgDisplay(iRow, iRow, 1) = 0;
    mImgDisplay(iRow, iRow, 2) = 1;
    mImgDisplay(iRow, iRow, 3) = 0;
end

hFigure         = figure();
set(hFigure, 'Units', 'pixels', 'Position', [100, 100, 500, 500]);
hAxes           = axes('Units', 'pixels', 'Position', [50, 50, numCols, numRows]);
hImageObject    = image(mImgDisplay);
set(get(hAxes, 'Title'), 'String', 'Guided Filtered Image', 'FontSize', titleFontSize);

% Extract Line 01
vLine1 = mGuidedFilteredImage(refLine1Row, vRefLine1ColIdx);
% Extract Line 02
vLine2 = zeros([numRows, 1]);
for iRow = 1:numRows
    vLine2(iRow) = mGuidedFilteredImage(iRow, iRow);
end

hFigure = figure();
hAxes   = axes();
hLineSeries = plot([1:numCols], vLine1, [1:numRows], vLine2);
set(hLineSeries(1), 'LineWidth', normalLineWidth, 'Color', 'r');
set(hLineSeries(2), 'LineWidth', normalLineWidth, 'Color', 'g');
set(get(hAxes, 'Title'), 'String', 'Values Across Lines - Guided Filtered Image', 'FontSize', titleFontSize);
hLegend = legend({['Line #01'], ['Line #02']});
set(hLegend, 'FontSize', axisFotnSize);


%% Restore Defaults
set(0, 'DefaultFigureWindowStyle', 'normal');
set(0, 'DefaultAxesLooseInset', defaultLooseInset);

