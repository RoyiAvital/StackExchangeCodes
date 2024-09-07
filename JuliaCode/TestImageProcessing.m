% Test Image Processing
% Generates files for verification of algorithms in
% `JuliaImageProcessing.jl`.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     07/09/2024
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Simulation Constants

% Match Julia
PAD_MODE_CIRCULAR   = 0;
PAD_MODE_CONSTANT   = 1;
PAD_MODE_REFLECT    = 2;
PAD_MODE_REPLICATE  = 3;
PAD_MODE_SYMMETRIC  = 4;

% Match Julia
CONV_MODE_FULL  = 0;
CONV_MODE_SAME  = 1;
CONV_MODE_VALID = 2;


%% Simulation Parameters

numRows = 26;
numCols = 25;

repFactor = 15;

vPadMode    = [PAD_MODE_CIRCULAR; PAD_MODE_CONSTANT; PAD_MODE_REPLICATE; PAD_MODE_SYMMETRIC];
vConvMode   = [CONV_MODE_FULL; CONV_MODE_SAME; CONV_MODE_VALID];

vPadMode = repmat(vPadMode, repFactor, 1);
vConvMode = repmat(vConvMode, repFactor, 1);


%% Generate / Load Data

mI = rand(numRows, numCols);


%% Pad Array

cPad = {length(vPadMode), 3}; %<! Output, vPadSize, PadMode

for ii = 1:length(vPadMode)
    vPadSize = randi([1, 9], 1, 2);
    switch(vPadMode(ii))
        case(PAD_MODE_CIRCULAR)
            padMode = 'circular';
        case(PAD_MODE_CONSTANT)
            padMode = 'constant';
        case(PAD_MODE_REPLICATE)
            padMode = 'replicate';
        case(PAD_MODE_SYMMETRIC)
            padMode = 'symmetric';
    end
    if (vPadMode(ii) == PAD_MODE_CONSTANT)
        mIPad = padarray(mI, vPadSize, 0, 'both');
    else
        mIPad = padarray(mI, vPadSize, padMode, 'both');
    end
    cPad{ii, 1} = mIPad;
    cPad{ii, 2} = vPadSize;
    cPad{ii, 3} = padMode;
end


%% Convolution

cConv = {length(vConvMode), 4}; %<! Output, vKerSize, mK, convMode

for ii = 1:length(vConvMode)
    vKerSize = randi([1, 9], 1, 2);
    mK = rand(vKerSize); %<! Must be row vector
    switch(vConvMode(ii))
        case(CONV_MODE_FULL)
            convMode = 'full';
        case(CONV_MODE_SAME)
            convMode = 'same';
        case(CONV_MODE_VALID)
            convMode = 'valid';
    end
    mIConv = conv2(mI, mK, convMode);
    cConv{ii, 1} = mIConv;
    cConv{ii, 2} = vKerSize;
    cConv{ii, 3} = mK;
    cConv{ii, 4} = convMode;
end


%% Save Data

save('TestImageProcessing', 'mI', 'cPad', 'cConv');



%% Auxiliary Functions



%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

