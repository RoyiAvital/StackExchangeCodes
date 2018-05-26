% StackExchange Signal Processing Q19646
% https://dsp.stackexchange.com/questions/19646/
% Are Convolution and Deconvolution Kernels the Same?
% References:
%   1.  aa
% Remarks:
%   1.  This creates a Convolution Kernel which is the inverse of itself.
%       Namely in the case of Deconvolution the Convolution Kernel and the
%       Deconvolution Kernel are the same.
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     26/05/2018  Royi
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numElements = 11; %<! Number of Elements in the Kernel


%% Generating the Kernel (Real)

isEven = mod(numElements, 2) == 0;
numElmntsBase = ceil((numElements + isEven) / 2);

% Base Kernel (Before Complex Conjugate Symmetry) must contain -1 or 1.
vBaseKernel = round(rand([numElmntsBase, 1]));
vBaseKernel(vBaseKernel == 0) = -1;

% Complex Conjugate Symmetry (Pay attention to Odd / Even number of
% elements)
vConvKernelDft = [vBaseKernel; flip(vBaseKernel(2:(end - isEven)), 1)];

vConvKernel = ifft(vConvKernelDft); %<! The Convolution Kernel in Time Domain


%% Display Results

figureIdx = figureIdx + 1;

hFigure         = figure('Position', figPosLarge);
hAxes           = subplot(3, 1, 1);
hLineSeries     = line(1:numElements, vConvKernelDft);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Convolution Kernel Elements - Fourier Domain']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Element Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Element Value']}, ...
    'FontSize', fontSizeAxis);

hAxes           = subplot(3, 1, 2);
hLineSeries     = line(1:numElements, vConvKernel);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Convolution Kernel Elements - Time Domain']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Element Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Element Value']}, ...
    'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['Ground Truth'], ['Estimated']});

hAxes           = subplot(3, 1, 3);
hLineSeries     = line(1:numElements, cconv(vConvKernel, vConvKernel, numElements));
% hLineSeries     = line(1:numElements, ifft(fft(vConvKernel) .* fft(vConvKernel))); %<! Circular Convolution in the Frequency Domain
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Circular Convolution of the Kernel with Itself - Time Domain']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Element Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Element Value']}, ...
    'FontSize', fontSizeAxis);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

