
OPERATION_MODE_CONVOLUTION = 1;
OPERATION_MODE_CORRELATION = 2;

CONVOLUTION_SHAPE_FULL  = 1;
CONVOLUTION_SHAPE_SAME  = 2;
CONVOLUTION_SHAPE_VALID = 3;

numRowsImage = 6;
numColsImage = 7;

numRowsKernel = 2;
numColsKernel = 4;

opMode = OPERATION_MODE_CONVOLUTION;

mA = rand(numRowsImage, numColsImage);
mH = reshape(1:(numRowsKernel * numColsKernel), [numRowsKernel, numColsKernel]);

%% Full

convShape   = CONVOLUTION_SHAPE_VALID;
numElements = numRowsImage;
mK = full(CreateImageConvMtx(mH, numRowsImage, numColsImage, convShape));
mKK = CreateConvMtx(mH(:, 1), numElements, opMode, convShape);
mKKK = full(CreateConvMtxSparse(mH(:, 1), numElements, opMode, convShape));
mKKKK = full(CreateImageConvMtxSparse(mH, numRowsImage, numColsImage, convShape));


isequal(mK, mKKKK)






