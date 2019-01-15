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
        mK = CreateConvMtxZeros(mH, numRows, numCols);
    case(BOUNDARY_MODE_SYMMETRIC)
        mK = CreateConvMtxSymmetric(mH, numRows, numCols);
    case(BOUNDARY_MODE_REPLICATE)
        mK = CreateConvMtxReplicate(mH, numRows, numCols);
    case(BOUNDARY_MODE_CIRCULAR)
        mK = CreateConvMtxCircular(mH, numRows, numCols);
end


end


function [ mK ] = CreateConvMtxZeros( mH, numRows, numCols )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

numElementsImage    = numRows * numCols;
numRowsKernel       = size(mH, 1);
numColsKernel       = size(mH, 2);
numElementsKernel   = numRowsKernel * numColsKernel;

vRows = reshape(repmat(1:numElementsImage, numElementsKernel, 1), numElementsImage * numElementsKernel, 1);
vCols = zeros(numElementsImage * numElementsKernel, 1);
vVals = zeros(numElementsImage * numElementsKernel, 1);

kernelRadiusV = floor(numRowsKernel / 2);
kernelRadiusH = floor(numColsKernel / 2);

pxIdx       = 0;
elmntIdx    = 0;

for jj = 1:numCols
    for ii = 1:numRows
        pxIdx = pxIdx + 1;
        for ll = -kernelRadiusH:kernelRadiusH
            for kk = -kernelRadiusV:kernelRadiusV
                elmntIdx = elmntIdx + 1;
                
                pxShift = (ll * numCols) + kk;
                
                if((ii + kk <= numRows) && (ii + kk >= 1) && (jj + ll <= numCols) && (jj + ll >= 1))
                    vCols(elmntIdx) = pxIdx + pxShift;
                    vVals(elmntIdx) = mH(kk + kernelRadiusV + 1, ll + kernelRadiusH + 1);
                else
                    vCols(elmntIdx) = pxIdx;
                    vVals(elmntIdx) = 0; % See the accumulation property of 'sparse()'.
                end
            end
        end
    end
end

mK = sparse(vRows, vCols, vVals, numElementsImage, numElementsImage);


end


function [ mK ] = CreateConvMtxSymmetric( mH, numRows, numCols )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

numElementsImage    = numRows * numCols;
numRowsKernel       = size(mH, 1);
numColsKernel       = size(mH, 2);
numElementsKernel   = numRowsKernel * numColsKernel;

vRows = reshape(repmat(1:numElementsImage, numElementsKernel, 1), numElementsImage * numElementsKernel, 1);
vCols = zeros(numElementsImage * numElementsKernel, 1);
vVals = zeros(numElementsImage * numElementsKernel, 1);

kernelRadiusV = floor(numRowsKernel / 2);
kernelRadiusH = floor(numColsKernel / 2);

pxIdx       = 0;
elmntIdx    = 0;

for jj = 1:numCols
    for ii = 1:numRows
        pxIdx = pxIdx + 1;
        for ll = -kernelRadiusH:kernelRadiusH
            for kk = -kernelRadiusV:kernelRadiusV
                elmntIdx = elmntIdx + 1;
                
                pxShift = (ll * numCols) + kk;
                
                if(ii + kk > numRows)
                    pxShift = pxShift - (2 * (ii + kk - numRows) - 1);
                end
                
                if(ii + kk < 1)
                    pxShift = pxShift + (2 * (1 -(ii + kk)) - 1);
                end
                
                if(jj + ll > numCols)
                    pxShift = pxShift - ((2 * (jj + ll - numCols) - 1) * numCols);
                end
                
                if(jj + ll < 1)
                    pxShift = pxShift + ((2 * (1 - (jj + ll)) - 1) * numCols);
                end
                
                vCols(elmntIdx) = pxIdx + pxShift;
                vVals(elmntIdx) = mH(kk + kernelRadiusV + 1, ll + kernelRadiusH + 1);
                
            end
        end
    end
end

mK = sparse(vRows, vCols, vVals, numElementsImage, numElementsImage);


end


function [ mK ] = CreateConvMtxReplicate( mH, numRows, numCols )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

numElementsImage    = numRows * numCols;
numRowsKernel       = size(mH, 1);
numColsKernel       = size(mH, 2);
numElementsKernel   = numRowsKernel * numColsKernel;

vRows = reshape(repmat(1:numElementsImage, numElementsKernel, 1), numElementsImage * numElementsKernel, 1);
vCols = zeros(numElementsImage * numElementsKernel, 1);
vVals = zeros(numElementsImage * numElementsKernel, 1);

kernelRadiusV = floor(numRowsKernel / 2);
kernelRadiusH = floor(numColsKernel / 2);

pxIdx       = 0;
elmntIdx    = 0;

for jj = 1:numCols
    for ii = 1:numRows
        pxIdx = pxIdx + 1;
        for ll = -kernelRadiusH:kernelRadiusH
            for kk = -kernelRadiusV:kernelRadiusV
                elmntIdx = elmntIdx + 1;
                
                pxShift = (ll * numCols) + kk;
                
                if(ii + kk > numRows)
                    pxShift = pxShift - (ii + kk - numRows);
                end
                
                if(ii + kk < 1)
                    pxShift = pxShift + (1 -(ii + kk));
                end
                
                if(jj + ll > numCols)
                    pxShift = pxShift - ((jj + ll - numCols) * numCols);
                end
                
                if(jj + ll < 1)
                    pxShift = pxShift + ((1 - (jj + ll)) * numCols);
                end
                
                vCols(elmntIdx) = pxIdx + pxShift;
                vVals(elmntIdx) = mH(kk + kernelRadiusV + 1, ll + kernelRadiusH + 1);
                
            end
        end
    end
end

mK = sparse(vRows, vCols, vVals, numElementsImage, numElementsImage);


end


function [ mK ] = CreateConvMtxCircular( mH, numRows, numCols )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

numElementsImage    = numRows * numCols;
numRowsKernel       = size(mH, 1);
numColsKernel       = size(mH, 2);
numElementsKernel   = numRowsKernel * numColsKernel;

vRows = reshape(repmat(1:numElementsImage, numElementsKernel, 1), numElementsImage * numElementsKernel, 1);
vCols = zeros(numElementsImage * numElementsKernel, 1);
vVals = zeros(numElementsImage * numElementsKernel, 1);

kernelRadiusV = floor(numRowsKernel / 2);
kernelRadiusH = floor(numColsKernel / 2);

pxIdx       = 0;
elmntIdx    = 0;

for jj = 1:numCols
    for ii = 1:numRows
        pxIdx = pxIdx + 1;
        for ll = -kernelRadiusH:kernelRadiusH
            for kk = -kernelRadiusV:kernelRadiusV
                elmntIdx = elmntIdx + 1;
                
                pxShift = (ll * numCols) + kk;
                
                if(ii + kk > numRows)
                    pxShift = pxShift - numRows;
                end
                
                if(ii + kk < 1)
                    pxShift = pxShift + numRows;
                end
                
                if(jj + ll > numCols)
                    pxShift = pxShift - (numCols * numCols);
                end
                
                if(jj + ll < 1)
                    pxShift = pxShift + (numCols * numCols);
                end
                
                vCols(elmntIdx) = pxIdx + pxShift;
                vVals(elmntIdx) = mH(kk + kernelRadiusV + 1, ll + kernelRadiusH + 1);
                
            end
        end
    end
end

mK = sparse(vRows, vCols, vVals, numElementsImage, numElementsImage);


end

