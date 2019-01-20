function [ mK ] = CreateImageConvMtxSparse( mH, numRows, numCols, convShape )
% ----------------------------------------------------------------------------------------------- %
% [ mK ] = CreateImageConvMtx( mH, numRows, numCols, convShape )
% Generates a Convolution Matrix for the 2D Kernel (The Matrix mH) with
% support for different convolution shapes (Full / Same / Valid).
% Input:
%   - mH                -   Input 2D Convolution Kernel.
%                           Structure: Matrix.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - numRows           -   Number of Rows.
%                           Number of rows in the output convolution
%                           matrix.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, ...}.
%   - numCols           -   Number of Columns.
%                           Number of columns in the output convolution
%                           matrix.
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
%                           The output convolution matrix. Multiplying in
%                           the column stack form on an image should be
%                           equivalent to applying convolution on the
%                           image.
%                           Structure: Matrix (Sparse).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  MATLAB's 'convmtx2()' - https://www.mathworks.com/help/images/ref/convmtx2.html.
%   2.  Matt J Maethod - https://www.mathworks.com/matlabcentral/answers/439928#answer_356557.
% Remarks:
%   1.  This method builds the Impulse Response per pixel location for the
%       output matrix. Basically, each column of the 'mK' matrix is the
%       impulese response to the pixel at the i-th location.
% TODO:
%   1.  
%   Release Notes:
%   -   1.0.001     17/01/2018  Royi Avital
%       *   Faster concatenation of the sparase matrix (Without calling
%           'struct2mat()').
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
    cBlockMtx{ii} = CreateConvMtxSparse(mH(:, ii), numRows, convShape);
    % cBlockMtx{ii} = full(CreateConvMtxSparse(mH(:, ii), numRows, operationMode, convShape));
end

switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        diagIdx     = 0;
        numRowsKron = numCols + numColsKernel - 1;
    case(CONVOLUTION_SHAPE_SAME)
        diagIdx     = floor(numColsKernel / 2);
        numRowsKron = numCols;
    case(CONVOLUTION_SHAPE_VALID)
        diagIdx     = numColsKernel - 1;
        numRowsKron = numCols - numColsKernel + 1;
end

mK = kron(spdiags(ones(numel(cBlockMtx{1})), diagIdx, numRowsKron, numCols), cBlockMtx{1});
% mK = kron(full(spdiags(ones(numel(cBlockMtx{1})), diagIdx, numRowsKron, numCols)), cBlockMtx{1});
for ii = 2:numBlockMtx
    diagIdx = diagIdx - 1;
    mK = mK + kron(spdiags(ones(numel(cBlockMtx{1})), diagIdx, numRowsKron, numCols), cBlockMtx{ii});
    % mK = mK + kron(full(spdiags(ones(numel(cBlockMtx{1})), diagIdx, numRowsKron, numCols)), cBlockMtx{ii});
end


end

