% StackExchange Signal Processing Q38542
% https://dsp.stackexchange.com/questions/38542
% Applying Image Filtering (Circular Convolution) in Frequency Domain
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes:
% - 1.0.000     15/03/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

imageFileName           = 'Lena.png';
gaussianKernelStd       = 4;
gaussianKernelRadius    = ceil(5 * gaussianKernelStd);


%% Load / Generate Data

% Load Image
mI = im2single(imread(imageFileName));
numRows = size(mI, 1);
numCols = size(mI, 2);

% Generate the Gaussian Blur Kernel
% Pay attention, the separability property of the Gaussian Kernel is
% ignored in this case. It is used as general 2D Kernel.
vX = [-gaussianKernelRadius:gaussianKernelRadius].';
vK = exp(-(vX .* vX) ./ (2 * gaussianKernelStd * gaussianKernelStd));
vK = vK ./ sum(vK);
mK = vK * vK.';

% The image (0, 0) is at top left. Hence we generate a kernel with the same
% size as the image and its (0, ) at the top left.
mKC = CircularExtension2D(mK, numRows, numCols);


%% Analysis

mIPad = padarray(mI, [gaussianKernelRadius, gaussianKernelRadius], 'circular', 'both');
mORef = conv2(mIPad, mK, 'valid');

% Frequency Domain Convolution
% Data is real hence 'ifft()' can be used in Symmetric Mode
mO = ifft2(fft2(mKC) .* fft2(mI), 'symmetric');


%% Display Results

disp([' ']);
disp(['Maximum Absolute Error - ', num2str(max(abs(mO(:) - mORef(:))))]);
disp([' ']);

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosX2Large);
hAxes       = subplot(2, 2, 1);
hSurfObj    = surf(vX, vX, mK);
set(get(hAxes, 'Title'), 'String', {['Gaussian Kernel'], ['Kernel STD - ', num2str(gaussianKernelStd)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['x']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['y']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'ZLabel'), 'String', {['z']}, ...
    'FontSize', fontSizeAxis);

hAxes       = subplot(2, 2, 2);
hImageObj   = imshow(mI);
set(get(hAxes, 'Title'), 'String', {['Input Image - Lena']}, ...
    'FontSize', fontSizeTitle);

hAxes   = subplot(2, 2, 3);
hImageObj   = imshow(mORef);
set(get(hAxes, 'Title'), 'String', {['Reference Output Image - Spatial Circular Convolution']}, ...
    'FontSize', fontSizeTitle);

hAxes   = subplot(2, 2, 4);
hImageObj   = imshow(mO);
set(get(hAxes, 'Title'), 'String', {['Output Image - Frequency Domain Convolution'], ...
    ['Maximum Absolute Error - ', num2str(max(abs(mO(:) - mORef(:))))]}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

