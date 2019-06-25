% StackExchange Signal Processing Q59089
% https://dsp.stackexchange.com/questions/59089
% The Gradient of Least Squares of 2D Image Convolution
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     24/06/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;

SHAPE_MODE_FULL   = 'full';
SHAPE_MODE_VALID  = 'valid';

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;
DIFF_MODE_COMPLEX   = 4;


%% Simulation Parameters

numRowsImage = 13;
numColsImage = 10;

numRowsKernel = 5;
numColsKernel = 3;

diffMode    = DIFF_MODE_COMPLEX;
epsVal      = 1e-6;

hCorr2D = @(mX, mH, convShape) conv2(mX, flip(flip(mH, 1), 2), convShape);

%% Generate Data

mX = randn(numRowsImage, numColsImage);
mH = randn(numRowsKernel, numColsKernel);
mY = randn(numRowsImage - numRowsKernel + 1, numColsImage - numColsKernel + 1);

hConvX = @(vX) 0.5 * sum((conv2(reshape(vX, numRowsImage, numColsImage), mH, SHAPE_MODE_VALID) - mY) .^ 2, 'all');
hConvH = @(vH) 0.5 * sum((conv2(mX, reshape(vH, numRowsKernel, numColsKernel), SHAPE_MODE_VALID) - mY) .^ 2, 'all');


%% Gradient of Convolution

% Gradient with respect to X
vGNumeric   = CalcFunGrad(mX(:), hConvX, diffMode, epsVal);
vGAnalytic  = conv2((conv2(mX, mH, SHAPE_MODE_VALID) - mY), mH(end:-1:1, end:-1:1), SHAPE_MODE_FULL); 
% vGAnalytic = hCorr2D((conv2(mX, mH, SHAPE_MODE_VALID) - mY), mH, SHAPE_MODE_FULL);

max(abs(vGNumeric(:) - vGAnalytic(:)))

% Gradient with respect to H
vGNumeric   = CalcFunGrad(mH(:), hConvH, diffMode, epsVal);
vGAnalytic  = conv2(mX(end:-1:1, end:-1:1), (conv2(mX, mH, SHAPE_MODE_VALID) - mY), SHAPE_MODE_VALID);

max(abs(vGNumeric(:) - vGAnalytic(:)))

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

