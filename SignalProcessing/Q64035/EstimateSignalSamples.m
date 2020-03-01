function [ vX ] = EstimateSignalSamples( vX, vH, vY, convShape, paramLambda, hSumLogProb )
% ----------------------------------------------------------------------------------------------- %
% [ mK ] = CreateConvMtx1D( vK, numElements, convShape )
% Generates a Convolution Matrix for 1D Kernel (The Vector vK) with
% support for different convolution shapes (Full / Same / Valid). The
% matrix is build such that for a signal 'vS' with 'numElements = size(vS
% ,1)' the following are equiavlent: 'mK * vS' and conv(vS, vK,
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
%   2.  In caes the same convolution is applied on many vectors, stacking
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

switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        convShapeString = 'full';
    case(CONVOLUTION_SHAPE_SAME)
        convShapeString = 'same';
    case(CONVOLUTION_SHAPE_VALID)
        convShapeString = 'valid';
end

vXX = vX;

hObjFun = @(vX) 0.5 * sum((conv(vX, vH, convShapeString) - vY) .^ 2) - (paramLambda * hSumLogProb(vX));

vX = fminunc(hObjFun, vX);


end

