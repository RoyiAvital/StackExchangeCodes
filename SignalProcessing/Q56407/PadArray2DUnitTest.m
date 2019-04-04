% StackExchange Signal Processing Q56407
% https://dsp.stackexchange.com/questions/56407
% How Much Zero Padding Do We Need to Perform Filtering in the Fourier Domain?
% References:
%   1.  A
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

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;

PADDING_MODE_ZEROS      = 1;
PADDING_MODE_SYMMETRIC  = 2;
PADDING_MODE_REPLICATE  = 3;
PADDING_MODE_CIRCULAR   = 4;


%% Simulation Parameters

numRows = 10;
numCols = 10;

maxRadius = 5;

numTests = 1000;


%% Generate Data

mI = randi([-10, 10], [numRows, numCols]);


%% Unit Test

for ii = 1:numTests
    paddingMode = randi([1, 4], [1, 1]);
    vPadRadius = randi([1, maxRadius], [2, 1]);
    
    mO = PadArray2D(mI, vPadRadius, paddingMode);
    
    switch(paddingMode)
        case(PADDING_MODE_ZEROS)
            mORef = padarray(mI, vPadRadius, 0, 'both');
        case(PADDING_MODE_SYMMETRIC)
            mORef = padarray(mI, vPadRadius, 'symmetric', 'both');
        case(PADDING_MODE_REPLICATE)
            mORef = padarray(mI, vPadRadius, 'replicate', 'both');
        case(PADDING_MODE_CIRCULAR)
            mORef = padarray(mI, vPadRadius, 'circular', 'both');
    end
    
    if(any(mORef(:) ~= mO(:)))
        keyboard;
    end
    
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

