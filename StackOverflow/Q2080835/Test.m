
OPERATION_MODE_CONVOLUTION = 1;
OPERATION_MODE_CORRELATION = 2;

BOUNDARY_MODE_ZEROS         = 1;
BOUNDARY_MODE_SYMMETRIC     = 2;
BOUNDARY_MODE_REPLICATE     = 3;
BOUNDARY_MODE_CIRCULAR      = 4;

operationMode   = OPERATION_MODE_CONVOLUTION;
boundaryMode    = BOUNDARY_MODE_CIRCULAR;

switch(operationMode)
    case(OPERATION_MODE_CONVOLUTION)
        operationModeString = 'conv';
    case(OPERATION_MODE_CORRELATION)
        operationModeString = 'corr';
end

switch(boundaryMode)
    case(BOUNDARY_MODE_ZEROS)
        boundaryModeString = 0;
    case(BOUNDARY_MODE_SYMMETRIC)
        boundaryModeString = 'symmetric';
    case(BOUNDARY_MODE_REPLICATE)
        boundaryModeString = 'replicate';
    case(BOUNDARY_MODE_CIRCULAR)
        boundaryModeString = 'circular';
end


numRows = 15;
numCols = 15;

numRowsKernel = 5;
numColsKernel = 3;

mI = rand(numRows, numCols);
mH = rand(numRowsKernel, numColsKernel);

mORef = imfilter(mI, mH, boundaryModeString, 'same', operationModeString);

mK = CreateImageFilterMtx(mH, numRows, numCols, operationMode, boundaryMode);
mO = reshape(mK * mI(:), numRows, numCols);

mE = mO - mORef;
max(abs(mE(:)))


