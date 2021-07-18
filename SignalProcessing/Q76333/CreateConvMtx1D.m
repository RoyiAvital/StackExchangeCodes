function [ mK ] = CreateConvMtx1D( vK, numElements, convShape )
% ----------------------------------------------------------------------------------------------- %
% [ mK ] = CreateConvMtx1D( vK, numElements, convShape )
% Generates a Convolution Matrix for 1D Kernel (The Vector vK) with
% support for different convolution shapes (Full / Same / Valid). The
% matrix is build such that for a signal 'vS' with 'numElements = size(vS
% ,1)' the following are equivalent: 'mK * vS' and conv(vS, vK,
% convShapeString);
% Input:
%   - vK                -   Input 1D Convolution Kernel.
%                           Structure: Vector.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - numElements       -   Number of Elements.
%                           Number of elements of the vector to be
%                           convolved with the matrix. Basically set the
%                           number of columns of the Convolution Matrix.
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
%                           'mK' and a vector 'vS' ('mK * vS') is the
%                           convolution between 'vK' and 'vS' with the
%                           corresponding convolution shape.
%                           Structure: Matrix (Sparse).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  MATLAB's 'convmtx()' - https://www.mathworks.com/help/signal/ref/convmtx.html.
% Remarks:
%   1.  The output matrix is sparse data type in order to make the
%       multiplication by vectors to more efficient.
%   2.  In case the same convolution is applied on many vectors, stacking
%       them into a matrix (Each signal as a vector) and applying
%       convolution on each column by matrix multiplication might be more
%       efficient than applying classic convolution per column.
% TODO:
%   1.  
%   Release Notes:
%   -   1.0.000     20/01/2019  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;

kernelLength    = length(vK);

switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        rowIdxFirst = 1;
        rowIdxLast  = numElements + kernelLength - 1;
        outputSize  = numElements + kernelLength - 1;
    case(CONVOLUTION_SHAPE_SAME)
        rowIdxFirst = 1 + floor(kernelLength / 2);
        rowIdxLast  = rowIdxFirst + numElements - 1;
        outputSize  = numElements;
    case(CONVOLUTION_SHAPE_VALID)
        rowIdxFirst = kernelLength;
        rowIdxLast  = (numElements + kernelLength - 1) - kernelLength + 1;
        outputSize  = numElements - kernelLength + 1;
end

mtxIdx = 0;

% The sparse matrix constructor ignores values of zero yet the Row / Column
% indices must be valid indices (Positive integers). Hence 'vI' and 'vJ'
% are initialized to 1 yet for invalid indices 'vV' will be 0 hence it has
% no effect.
vI = ones(numElements * kernelLength, 1);
vJ = ones(numElements * kernelLength, 1);
vV = zeros(numElements * kernelLength, 1);

for jj = 1:numElements
    for ii = 1:kernelLength
        if((ii + jj - 1 >= rowIdxFirst) && (ii + jj - 1 <= rowIdxLast))
            % Valid otuput matrix row index
            mtxIdx = mtxIdx + 1;
            vI(mtxIdx) = ii + jj - rowIdxFirst;
            vJ(mtxIdx) = jj;
            vV(mtxIdx) = vK(ii);
        end
    end
end

mK = sparse(vI, vJ, vV, outputSize, numElements);


end

