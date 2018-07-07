function [ mK ] = CreateImageFilterMtx( vK, numElements, operationMode, boundaryMode )
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
        vK = vK(end:-1:1).';
    case(OPERATION_MODE_CORRELATION)
        vK = vK.'; %<! Default Code is correlation
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


function [ mK ] = CreateConvMtxZeros( vK, numElements )
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
end


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

