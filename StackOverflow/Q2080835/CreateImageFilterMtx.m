function [ mK ] = CreateImageFilterMtx( mH, numRows, numCols, operationMode, boundaryMode )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

OPERATION_MODE_CONVOLUTION = 1;
OPERATION_MODE_CORRELATION = 2;

BOUNDARY_MODE_ZEROS         = 1;
BOUNDARY_MODE_SYMMETRIC     = 2;
BOUNDARY_MODE_REPLICATE     = 3;
BOUNDARY_MODE_CIRCULAR      = 4;

switch(operationMode)
    case(OPERATION_MODE_CONVOLUTION)
        mH = mH(end:-1:1, end:-1:1);
    case(OPERATION_MODE_CORRELATION)
        mH = mH; %<! Default Code is correlation
end

switch(boundaryMode)
    case(BOUNDARY_MODE_ZEROS)
        mK = CreateConvMtxZeros(vK, numElements);
    case(BOUNDARY_MODE_SYMMETRIC)
        mK = CreateConvMtxSymmetric(vK, numElements);
    case(BOUNDARY_MODE_REPLICATE)
        mK = CreateConvMtxReplicate(vK, numElements);
    case(BOUNDARY_MODE_CIRCULAR)
        mK = CreateConvMtxCircular(vK, numElements);
end


end


function [ mK ] = CreateConvMtxZeros( mH, numRows, numCols )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

numElementsImage    = numRows * numCols;
numElementsKernel   = numel(mH);

vRows = reshape(repmat(1:numElementsImage, numElementsKernel, 1), numElementsImage * numElementsKernel, 1);
vCols = zeros(numRows * numCols * numElementsKernel, 1);
vVals = zeros(numRows * numCols * numElementsKernel, 1);

vShift = [1:numElementsKernel].' - ceil(numElementsKernel / 2);

vCurrCols = zeros(numElementsKernel, 1);
vCurrVals = zeros(numElementsKernel, 1);

iterIdx = 0;

pxIdx = 0;
colIdx = 0;

for jj = 1:numCols
    for ii = 1:numRows
        pxIdx = pxIdx + 1;
        for ll = 1:numColsKernel
            for kk = 1:numRowsKernel
                fd
            end
        end
    end
end




for ii = 1:numElementsImage
    
    vCurrCols(:) = ii + vShift;
    vCurrVals(:) = mH(:);
    
    vInvalidIdx = vCurrCols < 1;
    
    if(any(vInvalidIdx))
        vCurrVals(vInvalidIdx) = 0;
        vCurrCols(vInvalidIdx) = 1;
    end
    
    vCols(((iterIdx - 1) * numElementsKernel + 1):(iterIdx * numElementsKernel)) = vCurrCols;
    vVals(((iterIdx - 1) * numElementsKernel + 1):(iterIdx * numElementsKernel)) = vCurrVals;
end

mK = sparse(vRows, vCols, vVals, numRows * numCols, numRows * numCols);


end


function [ mK ] = CreateConvMtxSymmetric( vK, numElements )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

kernelLength    = length(vK);
kernelRadius    = floor(kernelLength / 2); %<! Assuming Odd Kernel
refIdx          = kernelRadius + 1; %<! Assuming Odd Kernel

mK = zeros([numElements, numElements]);

for ii = 1:numElements
    
    colFirstIdx  = max(ii - kernelRadius, 1);
    colLastIdx   = min(ii + kernelRadius, numElements);
    
    kernelFirstIdx    = colFirstIdx - ii + refIdx;
    kernelLastIdx     = colLastIdx - ii + refIdx;
    
    mK(ii, colFirstIdx:colLastIdx)   = vK(kernelFirstIdx:kernelLastIdx);
    
    if(kernelFirstIdx > 1)
        numCoeff = kernelRadius - ii + 1;
        mK(ii, 1:numCoeff) = mK(ii, 1:numCoeff) + vK(numCoeff:-1:1);
    end
    
    if(kernelLastIdx < kernelLength)
        numCoeff = ii + kernelRadius - numElements;
        mK(ii, numElements - numCoeff + 1:numElements) = mK(ii, numElements - numCoeff + 1:numElements) + vK(end:-1:end - numCoeff + 1);
    end
end


end


function [ mK ] = CreateConvMtxReplicate( vK, numElements )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

kernelLength    = length(vK);
kernelRadius    = floor(kernelLength / 2); %<! Assuming Odd Kernel
refIdx          = kernelRadius + 1; %<! Assuming Odd Kernel

mK = zeros([numElements, numElements]);

for ii = 1:numElements
    
    colFirstIdx  = max(ii - kernelRadius, 1);
    colLastIdx   = min(ii + kernelRadius, numElements);
    
    kernelFirstIdx    = colFirstIdx - ii + refIdx;
    kernelLastIdx     = colLastIdx - ii + refIdx;
    
    mK(ii, colFirstIdx:colLastIdx)   = vK(kernelFirstIdx:kernelLastIdx);
    
    if(kernelFirstIdx > 1)
        mK(ii, 1) = sum(vK(1:kernelFirstIdx));
    end
    
    if(kernelLastIdx < kernelLength)
        mK(ii, numElements) = sum(vK(kernelLastIdx:kernelLength));
    end
end


end


function [ mK ] = CreateConvMtxCircular( vK, numElements )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

kernelLength    = length(vK);
kernelRadius    = floor(kernelLength / 2); %<! Assuming Odd Kernel
refIdx          = kernelRadius + 1; %<! Assuming Odd Kernel

mK = zeros([numElements, numElements]);

for ii = 1:numElements
    
    colFirstIdx  = max(ii - kernelRadius, 1);
    colLastIdx   = min(ii + kernelRadius, numElements);
    
    kernelFirstIdx    = colFirstIdx - ii + refIdx;
    kernelLastIdx     = colLastIdx - ii + refIdx;
    
    mK(ii, colFirstIdx:colLastIdx)   = vK(kernelFirstIdx:kernelLastIdx);
    
    if(kernelFirstIdx > 1)
        numCoeff = kernelRadius - ii + 1;
        mK(ii, end - numCoeff + 1:end) = vK(1:numCoeff);
    end
    
    if(kernelLastIdx < kernelLength)
        numCoeff = ii + kernelRadius - numElements;
        mK(ii, 1:numCoeff) = vK(end - numCoeff + 1:end);
    end
end


end

