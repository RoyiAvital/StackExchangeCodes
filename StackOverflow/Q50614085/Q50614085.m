% Stack Overflow Q50614085
% https://stackoverflow.com/questions/50614085
% Applying Low Pass and Laplace of Gaussian Filter in Frequency Domain
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     31/05/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

imageFileName       = 'Lena.png';
gaussianKernalStd   = 5;
stdToRadiusFctr     = 5; %<! Trnasformin STD to grid radius


%% Generate / Load Data

mI = imread('Lena.png');
mI = mean(im2single(mI), 3); %<! Single Channel Image

kernelRadius    = ceil(stdToRadiusFctr * stdToRadiusFctr);
vGrid           = [-kernelRadius:kernelRadius].';

vGaussianKernel = exp(-(vGrid .* vGrid) ./ (2 * gaussianKernalStd * gaussianKernalStd));
vGaussianKernel = vGaussianKernel ./ sum(vGaussianKernel); %<! LPF Filter -> DC Gain = 1

mGaussianKernel = vGaussianKernel * vGaussianKernel.';
mLog = (((vGrid .^ 2) + (vGrid.' .^ 2) - (2 * gaussianKernalStd * gaussianKernalStd)) / (gaussianKernalStd ^ 4)) .* mGaussianKernel;
mLog = mLog - mean(mLog(:)); %<! HPF Filter -> DC Gain = 0


hFigure = figure('Position', figPosLarge);

hAxes   = subplot(2, 2, 1);
hImgObj = imagesc(vGrid, vGrid, mGaussianKernel);
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
set(get(hAxes, 'Title'), 'String', {['2D Gaussian Kernel']}, ...
    'Fontsize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index [x]']}, ...
    'Fontsize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Index [y]']}, ...
    'Fontsize', fontSizeAxis);

hAxes   = subplot(2, 2, 2);
hImgObj = imagesc(vGrid, vGrid, mLog);
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
set(get(hAxes, 'Title'), 'String', {['2D Laplacian of Gaussian Kernel']}, ...
    'Fontsize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index [x]']}, ...
    'Fontsize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Index [y]']}, ...
    'Fontsize', fontSizeAxis);

hAxes = subplot(2, 2, 3);
hLineObj = plot(vGrid, mGaussianKernel(ceil(size(vGrid, 1) / 2), :));
set(get(hAxes, 'Title'), 'String', {['2D Gaussian Kernel - Horizontal']}, ...
    'Fontsize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index [x]']}, ...
    'Fontsize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Kernel Value']}, ...
    'Fontsize', fontSizeAxis);

hAxes = subplot(2, 2, 4);
hLineObj = plot(vGrid, mLog(ceil(size(vGrid, 1) / 2), :));
set(get(hAxes, 'Title'), 'String', {['2D Laplacian of Gaussian Kernel - Horizontal']}, ...
    'Fontsize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Sample Index [x]']}, ...
    'Fontsize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Kernel Value']}, ...
    'Fontsize', fontSizeAxis);


%% Process

numRows = size(mI, 1);
numCols = size(mI, 2);

% Convolution in Spatial Domain
% Padding for Cyclic Convolution
mOGaussianRef = conv2(PadArrayCircular(mI, kernelRadius), mGaussianKernel, 'valid');
mOLogRef = conv2(PadArrayCircular(mI, kernelRadius), mLog, 'valid');

% Convolution in Frequency Domain
% Padding and centering of the Kernel (To match Circular Boundary Conditions)
mGaussianKernel(numRows, numCols) = 0;
mGaussianKernel = circshift(mGaussianKernel, [-kernelRadius, -kernelRadius]);

mLog(numRows, numCols) = 0;
mLog = circshift(mLog, [-kernelRadius, -kernelRadius]);

mOGaussian  = ifft2(fft2(mI) .* fft2(mGaussianKernel), 'symmetric');
mOLog       = ifft2(fft2(mI) .* fft2(mLog), 'symmetric');

convErr = norm(mOGaussianRef(:) - mOGaussian(:), 'inf');
disp(['Gaussian Kernel - Cyclic Convolution Error (Infinity Norm) - ', num2str(convErr)]);

convErr = norm(mOLogRef(:) - mOLog(:), 'inf');
disp(['LoG Kernel - Convolution Error (Infinity Norm) - ', num2str(convErr)]);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

