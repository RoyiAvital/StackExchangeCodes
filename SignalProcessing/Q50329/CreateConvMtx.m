function [ mK ] = CreateConvMtx( vK, numElements, operationMode, convShape )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

OPERATION_MODE_CONVOLUTION = 1;
OPERATION_MODE_CORRELATION = 2;

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;

switch(operationMode)
    case(OPERATION_MODE_CONVOLUTION)
        vK = vK(end:-1:1);
    case(OPERATION_MODE_CORRELATION)
        % vK = vK; %<! Default Code is correlation
end

kernelLength    = length(vK);
mK = zeros([numElements + kernelLength - 1, numElements]);

for ii = 1:numElements + kernelLength - 1
    kernelLastIdx     = min(kernelLength, kernelLength + numElements - ii);
    kernelFirstIdx    = max(kernelLastIdx - ii + 1, 1);
    
    kernelEffLength = kernelLastIdx - kernelFirstIdx + 1;
    
    colLastIdx   = min(ii, numElements);
    colFirstIdx  = colLastIdx - kernelEffLength + 1;
    
    mK(ii, colFirstIdx:colLastIdx)   = vK(kernelFirstIdx:kernelLastIdx);
end

switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        % mK = mK;
    case(CONVOLUTION_SHAPE_SAME)
        rowIdxFirst = 1 + floor(kernelLength / 2);
        rowIdxLast  = rowIdxFirst + numElements - 1;
        mK = mK(rowIdxFirst:rowIdxLast, :);
    case(CONVOLUTION_SHAPE_VALID)
        mK = mK(kernelLength:end - kernelLength + 1, :);
end


end

