% StackExchange Signal Processing Q76128
% https://dsp.stackexchange.com/questions/76128
% Modern Method for 1D Signal Segmentation
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     04/04/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numSamplesSegment = 50;

numSegments = 25; %<! Number of Segments

sineAmp     = 1;
sineFreqFac = 3; %<! [Hz]

dcStd = 2;

samplingFreq = 50;

noiseStd = 0.1;


%% Generate Data

%{
We'll generate 2 kind of signals:

1. Sine signal which changes its phase.
2. DC signal which changes its amplitude each segment.
%}

numSamples  = numSamplesSegment * numSegments;
vT          = linspace(0, 1, numSamplesSegment + 1).';
vT(end)     = [];

mS = zeros(numSamples, 2);

segmentStartIdx = 1;
segmentEndIdx   = numSamplesSegment;

for ii = 1:numSegments
    sineFreq = 1 + (sineFreqFac * rand(1, 1));
    sinePhase   = pi * rand(1, 1);
    dcVal       = dcStd * randn(1, 1);
    
    mS(segmentStartIdx:segmentEndIdx, 1) = sineAmp * sin(2 * pi * sineFreq * vT + sinePhase) + (noiseStd * randn(numSamplesSegment, 1));
    mS(segmentStartIdx:segmentEndIdx, 2) = dcVal + (noiseStd * randn(numSamplesSegment, 1));
    
    segmentStartIdx = segmentStartIdx + numSamplesSegment;
    segmentEndIdx   = segmentEndIdx + numSamplesSegment;
end


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

