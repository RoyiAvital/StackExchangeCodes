% Signal Processing Q2969
% https://dsp.stackexchange.com/questions/questions/2969
% Deconvolution of 1D Signals Blurred by Gaussian Kernel
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     25/09/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

OPERATION_MODE_CONVOLUTION = 1;
OPERATION_MODE_CORRELATION = 2;

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Simulation Parameters

numSamplesX = 300; %<! X Data Number of Samples
numCoeff    = 31; %<! Filter Number of Coefficients
numSamplesY = numSamplesX - numCoeff + 1; %<! Y Data Number of Samples

noiseStd = 0.01; %<! Noise Standard Deviation

kernelStdMinVal     = 0.1;
kernelStdMaxVal     = 2;
kernelStdNumVals    = 21;

kernelStd   = 2.5;

numIterations   = 60;
stepSize        = 0.00075;


%% Condition Number Analysis

vX = randn([numSamplesX, 1]);

kernelRadius    = floor(numCoeff / 2);
vKernelGrid     = [-kernelRadius:kernelRadius].';

vKernelStd  = linspace(kernelStdMinVal, kernelStdMaxVal, kernelStdNumVals);
vCondNumber = zeros([kernelStdNumVals, 1]);
vResErrSvd  = zeros([kernelStdNumVals, 1]);
vResErr     = zeros([kernelStdNumVals, 1]);
mHH         = zeros([numCoeff, kernelStdNumVals]);
cStdString  = cell([kernelStdNumVals, 1]);

for ii = 1:kernelStdNumVals
    vH = exp(-(vKernelGrid .^ 2) / (2 * vKernelStd(ii) * vKernelStd(ii)));
    vH = vH / sum(vH);
    mHH(:, ii) = vH;
    mH = CreateConvMtx(vH, numSamplesX, OPERATION_MODE_CONVOLUTION, CONVOLUTION_SHAPE_VALID);
    
    % vY = conv2(vX, vH, 'valid') + (noiseStd * randn([numSamplesY, 1]));
    vY = (mH * vX) + (noiseStd * randn([numSamplesY, 1]));
    
    vResErrSvd(ii) = norm(pinv(mH) * vY - vX, 2);
    vResErr(ii) = norm(mH \ vY - vX, 2);
    
    % vCondNumber(ii) = cond(mH.' * mH);
    vCondNumber(ii) = cond(mH);
    cStdString{ii} = ['Kernel STD - ', num2str(vKernelStd(ii))];
end

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = line(vKernelStd, 20 * log10(vCondNumber));
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Operator Condition Number as Function Kernel STD']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Kernel STD']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Condition Number [dB]']}, ...
    'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['Ground Truth'], ['Filter Estimation']});

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = line(vKernelGrid, mHH);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Gaussian Blur Kernel']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Kernel Tap']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend(cStdString);


hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = line(vKernelStd, [vResErr, vResErrSvd]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Reconstruction Error'], ['Noise STD - ', num2str(noiseStd)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Kernel STD']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['$ \left\| \hat{x} - x \right\|_{2} $']}, ...
    'FontSize', fontSizeAxis, 'Interpreter', 'latex');
hLegend = ClickableLegend({['\\'], ['Pseudo Inverse']});

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = line([1:numSamplesX], [vX, pinv(mH) * vY]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Reconstructed Signal'], ['Noise STD - ', num2str(noiseStd)]}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Samples Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Samples Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['Pseudo Inverse Estimated Signal']});


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

