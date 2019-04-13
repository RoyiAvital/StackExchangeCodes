% StackExchange Signal Processing Q56407
% https://dsp.stackexchange.com/questions/56407
% How Much Zero Padding Do We Need to Perform Filtering in the Fourier Domain?
% References:
%   1.  See Applying Low Pass and Laplace of Gaussian Filter in Frequency Domain - https://stackoverflow.com/questions/50614085.
%       Under my solution, the script 'FreqDomainConv.m'.
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

PADDING_MODE_ZEROS      = 1;
PADDING_MODE_SYMMETRIC  = 2;
PADDING_MODE_REPLICATE  = 3;
PADDING_MODE_CIRCULAR   = 4;


%% Simulation Parameters

minNumRows = 10;
minNumCols = 10;
maxNumRows = 16;
maxNumCols = 16;

minKernelLength = 2;
maxKernelLength = 6;

numTests = 10000;


%% Generate Data

% mI = randi([-10, 10], [numRows, numCols]);


%% Unit Test

for ii = 1:numTests
    
    numRows = randi([minNumRows, maxNumRows], [1, 1]);
    numCols = randi([minNumCols, maxNumCols], [1, 1]);
    
    mI = randi([-10, 10], [numRows, numCols]);
    
    paddingMode = randi([1, 4], [1, 1]);
    vKernelDim = randi([minKernelLength, maxKernelLength], [1, 2]);
    
    mH = randn(vKernelDim);
    
    
    mO      = ImageFilteringFrequencyDomain(mI, mH, paddingMode);
    mORef   = ImageFilteringSpatialDomain(mI, mH, paddingMode);
    
    if((size(mO, 1) ~= numRows) || (size(mO, 2) ~= numCols))
        keyboard;
    end
    
    
    if(max(abs(mO(:) - mORef(:))) > 1e-8)
        keyboard;
    end
    
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

