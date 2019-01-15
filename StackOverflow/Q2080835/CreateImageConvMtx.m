function [ mK ] = CreateImageConvMtx( mH, numRows, numCols, operationMode, convShape )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

% Not Working! Work in Progress.

OPERATION_MODE_CONVOLUTION = 1;
OPERATION_MODE_CORRELATION = 2;

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;

switch(operationMode)
    case(OPERATION_MODE_CONVOLUTION)
        mH = mH(end:-1:1, end:-1:1);
    case(OPERATION_MODE_CORRELATION)
        % mH = mH; %<! Default Code is correlation
end


numElementsImage    = numRows * numCols;
numRowsKernel       = size(mH, 1);
numColsKernel       = size(mH, 2);
numElementsKernel   = numRowsKernel * numColsKernel;


vRows = zeros(numElementsImage * numElementsKernel, 1);
vCols = zeros(numElementsImage * numElementsKernel, 1);
vVals = zeros(numElementsImage * numElementsKernel, 1);





























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

