% StackExchange Signal Processing Q74803
% https://dsp.stackexchange.com/questions/74803
% Replicate MATLAB's `conv2()` in Frequency Domain
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     04/04/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

CONV_SHAPE_FULL     = 1;
CONV_SHAPE_SAME     = 2;
CONV_SHAPE_VALID    = 3;


%% Simulation Parameters

minNumRows = 11;
minNumCols = 11;
maxNumRows = 20;
maxNumCols = 20;

minKernelLength = 1;
maxKernelLength = 10;

numTests = 100000;


%% Generate Data

% mI = im2single(imread('cameraman.tif'));
% mI = mI(:, :, 1); %<! Simulation works on Single Channel image
% 
% mH = ones(6, 5) / 30;
% 
% mORef   = conv2(mI, mH, 'full');
% mO      = ImageConvFrequencyDomain(mI, mH, CONV_SHAPE_FULL);
% 
% abs(max(mO(:) - mORef(:)))


%% Unit Test

for ii = 1:numTests
    
    numRows = randi([minNumRows, maxNumRows], [1, 1]);
    numCols = randi([minNumCols, maxNumCols], [1, 1]);
    
    mI = randi([-10, 10], [numRows, numCols]);
    
    convShape = randi([1, 3], [1, 1]);
    vKernelDim = randi([minKernelLength, maxKernelLength], [1, 2]);
    
    mH = randn(vKernelDim);
    
    mO      = ImageConvFrequencyDomain(mI, mH, convShape);
    switch(convShape)
        case(CONV_SHAPE_FULL)
            mORef   = conv2(mI, mH, 'full');
        case(CONV_SHAPE_SAME)
            mORef   = conv2(mI, mH, 'same');
        case(CONV_SHAPE_VALID)
            mORef   = conv2(mI, mH, 'valid');
    end
    
    if(max(abs(mO(:) - mORef(:))) > 1e-8)
        keyboard;
    end
    
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

