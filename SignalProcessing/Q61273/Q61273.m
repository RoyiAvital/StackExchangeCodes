% StackExchange Signal Processing Q61273
% https://dsp.stackexchange.com/questions/61273
% Estimate Noise Variance Given Multiple Realization of the Same Image
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     15/10/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 179;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

inputImageFileName  = 'InputImage.png';

vNumRealizations    = 2:50;
noiseStd            = 0.1;


%% Generate Data

mI = im2double(imread(inputImageFileName));

numRows = size(mI, 1);
numCols = size(mI, 2);

tI              = mI + (noiseStd * randn(numRows, numCols, vNumRealizations(end)));
mMeanImage      = zeros(numRows, numCols);
tN              = zeros(numRows, numCols, vNumRealizations(end));
mEstNoiseStd    = zeros(length(vNumRealizations), 2);
vN              = zeros(numRows * numCols * vNumRealizations(end), 1);

for ii = 1:length(vNumRealizations)
    numRealizations             = vNumRealizations(ii);
    mMeanImage(:)               = mean(tI(:, :, 1:numRealizations), 3);
    tN(:, :, 1:numRealizations) = tI(:, :, 1:numRealizations) - mMeanImage;
    numNoiseSamples             = numRows * numCols * numRealizations;
    vN(1:numNoiseSamples)       = reshape(tN(:, :, 1:numRealizations), numNoiseSamples, 1); %<! In MATLAB R2018b and above can use 'all' to skip this
    vEstNoiseStd(ii, 1)         = sqrt(mean(vN(1:numNoiseSamples) .^ 2)); % std(vN);
    vEstNoiseStd(ii, 2)         = mean(reshape(std(tI(:, :, 1:numRealizations), 0, 3), numRows * numCols, 1));
end


%% Analysis

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineObj    = line(vNumRealizations, [vEstNoiseStd, noiseStd * ones(length(vNumRealizations), 1)]);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(hLineObj(end), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Estimation of the Noise STD as a Function of Number of Realizations']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Number of Realizations']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['STD']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Estimated STD - Method 1'], ['Estimated STD - Method 2'], ['Gorund Truth']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

