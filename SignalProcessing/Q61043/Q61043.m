% StackExchange Signal Processing Q61043
% https://dsp.stackexchange.com/questions/61043
% Estimating Convolution Kernel from Input and Output Images
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     19/08/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;
DIFF_MODE_COMPLEX   = 4;


%% Simulation Parameters

inputImageFileName  = 'InputImage.jpg';
outputImageFileName = 'OutputImage.jpg';

numRows             = 200;
numCols             = 200;
squareHalfLength    = 15;
kernelStd           = 4;


numIterations   = 3000;
stepSize        = 2e-6;
kernelRadius    = 25;

diffMode    = DIFF_MODE_CENTRAL;
epsVal      = 1e-6;


%% Generate Data

mI = im2double(imread(inputImageFileName));
mI = mean(mI, 3);
% Padding the Input Image to match the model of Valid Convolution
mI = PadArrayReplicate(mI, kernelRadius);

mO = im2double(imread(outputImageFileName));
mO = mean(mO, 3);


numRows = 200;
numCols = 200;
mI = zeros([numRows, numCols]);
mI(86:115, 86:115) = 1;

vK = [-kernelRadius:kernelRadius];
vK = exp(-0.5 * (vK .* vK) / (kernelStd * kernelStd));
mK = vK.' * vK;
mK = mK / sum(mK(:));

mO = conv2(mI, mK, 'valid');









kernelLength = (2 * kernelRadius) + 1;
mH = zeros(kernelLength, kernelLength);

vRmseVal = zeros(numIterations, 1);
hCalcMse = @(mH) sqrt(sum((reshape(conv2(mI, mH, 'valid'), [], 1) - mO(:)) .^ 2));

hObjFun = @(vH) 0.5 * sum((reshape(conv2(mI, reshape(vH, kernelLength, kernelLength), 'valid'), [], 1) - mO(:)) .^ 2);


%% Analysis

vRmseVal(1) = hCalcMse(mH);

% Solving a Problem - Gradient Desecnt
for ii = 2:numIterations
    vG              = conv2(mI(end:-1:1, end:-1:1), (conv2(mI, mH, 'valid') - mO), 'valid');
    % vGG             = CalcFunGrad(mH(:), hObjFun, diffMode, epsVal);
    mH              = mH - (stepSize * vG);
    vRmseVal(ii)    = hCalcMse(mH);
    disp(['Finished Iteration #', num2str(ii, '%04d'), ' Out of ', num2str(numIterations)]);
end


figureIdx = figureIdx + 1;

hFigure         = figure('Position', figPosLarge);
hAxes           = subplot(2, 3, 1);
hImageObj       = imshow(mI);
set(get(hAxes, 'Title'), 'String', {['Input Image']}, ...
    'FontSize', fontSizeTitle);

hAxes           = subplot(2, 3, 2);
hImageObj       = imshow(mK, []);
set(get(hAxes, 'Title'), 'String', {['Convolution Kernel']}, ...
    'FontSize', fontSizeTitle);

hAxes           = subplot(2, 3, 3);
hImageObj       = imshow(mO);
set(get(hAxes, 'Title'), 'String', {['Output Image']}, ...
    'FontSize', fontSizeTitle);

hAxes           = subplot(2, 3, 4);
hLineSeries  = line(1:numIterations, vRmseVal);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['RMSE']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Iteration Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['RMSE']}, ...
    'FontSize', fontSizeAxis);

hAxes           = subplot(2, 3, 5);
hImageObj       = imshow(mH, []);
set(get(hAxes, 'Title'), 'String', {['Estimated Kernel']}, ...
    'FontSize', fontSizeTitle);

hAxes           = subplot(2, 3, 6);
hImageObj       = imshow(conv2(mI, mH, 'valid'));
set(get(hAxes, 'Title'), 'String', {['Output Image of Estimated Kernel']}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

