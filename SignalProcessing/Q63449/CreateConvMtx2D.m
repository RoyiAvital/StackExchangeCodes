function [ mK ] = CreateConvMtx2D( mH, numRows, numCols, convShape )
% ----------------------------------------------------------------------------------------------- %
% [ mK ] = CreateConvMtx2D( mH, numRows, numCols, convShape )
% Generates a Convolution Matrix for the 2D Kernel (The Matrix mH) with
% support for different convolution shapes (Full / Same / Valid).
% Input:
%   - mH                -   Input 2D Convolution Kernel.
%                           Structure: Matrix.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - numRows           -   Number of Rows.
%                           Number of rows of the image to be convolved.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, ...}.
%   - numCols           -   Number of Columns.
%                           Number of columns of the image to be convolved.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, ...}.
%   - convShape         -   Convolution Shape.
%                           The shape of the convolution which the output
%                           convolution matrix should represent. The
%                           options should match MATLAB's conv2() function
%                           - Full / Same / Valid.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3}.
% Output:
%   - mK                -   Convolution Matrix.
%                           The output convolution matrix. The product of
%                           the matrix 'mK' and and image 'mI' in its
%                           column stack form ('mK * mI(:)') is equivalent
%                           to the convolution of 'mI' with the kernel 'mH'
%                           using the corresponding convolution shape
%                           ('conv2(mI, mH, convShapeString)').
%                           Structure: Matrix (Sparse).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  MATLAB's 'convmtx2()' - https://www.mathworks.com/help/images/ref/convmtx2.html.
% Remarks:
%   1.  In caes the same convolution is applied on many images, stacking
%       them into a matrix (Each image as columns stacked vector) and
%       applying convolution on each column by matrix multiplication might
%       be more efficient than applying classic convolution per image.
%   2.  The output matrux has the form of doubly block toeplitz matrix. The
%       convolution shape sets where the diagonal of the first column to
%       appear.
% TODO:
%   1.  
%   Release Notes:
%   -   1.0.001     22/01/2018  Royi Avital
%       *   Fixed issue with the creatioon of the sparse diagonal matrix.
%           The vector to initialize the diagonal was much bigger than
%           needed. Improved performance in the Unit Test by factor of 10.
%   -   1.0.000     16/01/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

CONVOLUTION_SHAPE_FULL  = 1;
CONVOLUTION_SHAPE_SAME  = 2;
CONVOLUTION_SHAPE_VALID = 3;

numColsKernel   = size(mH, 2);
numBlockMtx     = numColsKernel;

cBlockMtx = cell(numBlockMtx, 1);

for ii = 1:numBlockMtx
    cBlockMtx{ii} = CreateConvMtx1D(mH(:, ii), numRows, convShape);
end

switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        % For convolution shape - 'full' the Doubly Block Toeplitz Matrix
        % has the first column as its main diagonal.
        diagIdx     = 0;
        numRowsKron = numCols + numColsKernel - 1;
    case(CONVOLUTION_SHAPE_SAME)
        % For convolution shape - 'same' the Doubly Block Toeplitz Matrix
        % has the first column shifted by the kernel horizontal radius.
        diagIdx     = floor(numColsKernel / 2);
        numRowsKron = numCols;
    case(CONVOLUTION_SHAPE_VALID)
        % For convolution shape - 'valid' the Doubly Block Toeplitz Matrix
        % has the first column shifted by the kernel horizontal length.
        diagIdx     = numColsKernel - 1;
        numRowsKron = numCols - numColsKernel + 1;
end

vI = ones(min(numRowsKron, numCols), 1);
mK = kron(spdiags(vI, diagIdx, numRowsKron, numCols), cBlockMtx{1});
for ii = 2:numBlockMtx
    diagIdx = diagIdx - 1;
    mK = mK + kron(spdiags(vI, diagIdx, numRowsKron, numCols), cBlockMtx{ii});
end


end

