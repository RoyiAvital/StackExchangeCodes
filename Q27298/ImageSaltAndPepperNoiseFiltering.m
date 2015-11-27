% Image Salt and Pepper Noise Filtering
% See http://dsp.stackexchange.com/questions/27298

%% General Parameters and Initialization

clear();
close('all');

% set(0, 'DefaultFigureWindowStyle', 'docked');
defaultLooseInset = get(0, 'DefaultAxesLooseInset');
set(0, 'DefaultAxesLooseInset', [0.05, 0.05, 0.05, 0.05]);

titleFontSize   = 14;
axisFotnSize    = 12;
stringFontSize  = 12;

thinLineWidth   = 2;
normalLineWidth = 3;
thickLineWidth  = 4;

smallSizeData   = 36;
mediumSizeData  = 48;
bigSizeData     = 60;

randomNumberStream = RandStream('mlfg6331_64', 'NormalTransform', 'Ziggurat');
subStreamNumber = 57162;
set(randomNumberStream, 'Substream', subStreamNumber);
RandStream.setGlobalStream(randomNumberStream);

%% Constants
UNIT8_MIN_VALUE = 0;
UNIT8_MAX_VALUE = 255;


%% Loading Data

mInputImage = imread('../RawData/Image0001.png');
mInputImage = mInputImage(:, :, 1); %<! Assuring Single Channel Image

numRows = size(mInputImage, 1);
numCols = size(mInputImage, 2);
numPixels = numRows * numCols;


%% Parameters
winRadius           = 1; %<! Loacl Window Radius
noiseProbability    = 0.05; %<! Probability of a Pixel to be affected by noise


%% Generating Noisy Image

numNoisyPixels  = round(noiseProbability * numPixels);

vNoisyPixelsIdx         = randperm(numPixels, numNoisyPixels);
vPepperNoisyPixelsIdx   = vNoisyPixelsIdx(1:floor(numNoisyPixels / 2));
vSaltNoisyPixelsIdx     = vNoisyPixelsIdx((floor(numNoisyPixels / 2) + 1):numNoisyPixels);

mNoisyImage                         = mInputImage;
mNoisyImage(vPepperNoisyPixelsIdx)  = UNIT8_MAX_VALUE;
mNoisyImage(vSaltNoisyPixelsIdx)    = UNIT8_MIN_VALUE;


%% Image Median Filtering

mFilteredImage = mNoisyImage;

for iCol = 1:numCols
    for jRow = 1:numRows
        winFirstRowIdx  = max(1, (jRow - winRadius));
        winLastRowIdx   = min(numRows, (jRow + winRadius));
        winFirstColIdx  = max(1, (iCol - winRadius));
        winLastColIdx   = min(numCols, (iCol + winRadius));
        
        vRowsIdx = [winFirstRowIdx:winLastRowIdx];
        vColsIdx = [winFirstColIdx:winLastColIdx];
        
        mLocalWin = mFilteredImage(vRowsIdx, vColsIdx);
        
        mFilteredImage(jRow, iCol) = median(mLocalWin(:));
    end
end


%% Display Results

hFigure = figure('Position', [100, 100, 900, 550], 'Units', 'pixels');
set(hFigure, 'Colormap', gray(256));
hAxes   = axes();
set(hAxes, 'Units', 'pixels')
set(hAxes, 'Position', [50, 50, numCols, numRows]);
hImageObject = image(mInputImage);
set(get(hAxes, 'Title'), 'String', ['Input Image'], ...
    'FontSize', titleFontSize);

hFigure = figure('Position', [100, 100, 900, 550], 'Units', 'pixels');
set(hFigure, 'Colormap', gray(256));
hAxes   = axes();
set(hAxes, 'Units', 'pixels')
set(hAxes, 'Position', [50, 50, numCols, numRows]);
hImageObject = image(mNoisyImage);
set(get(hAxes, 'Title'), 'String', ['Noisy Image'], ...
    'FontSize', titleFontSize);

hFigure = figure('Position', [100, 100, 900, 550], 'Units', 'pixels');
set(hFigure, 'Colormap', gray(256));
hAxes   = axes();
set(hAxes, 'Units', 'pixels')
set(hAxes, 'Position', [50, 50, numCols, numRows]);
hImageObject = image(mFilteredImage);
set(get(hAxes, 'Title'), 'String', ['Filtered Image'], ...
    'FontSize', titleFontSize);



%% Restore Defaults
set(0, 'DefaultFigureWindowStyle', 'normal');
set(0, 'DefaultAxesLooseInset', defaultLooseInset);

