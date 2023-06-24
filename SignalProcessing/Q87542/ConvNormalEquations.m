function [ vY ] = ConvNormalEquations( vX, vH, vG, convShape, paramLambda )
% ----------------------------------------------------------------------------------------------- %
% [ vY ] = ConvNormalEquations( vX, vH, vG, convShape, paramLambda )
% Calculate the output of ((mH.' * mH) + paramLambda * (mG.' * mG)) * vX
% using the convolution kernels vH and vG where mH = CreateConvMtx1D( vH,
% length(vX), convShape ).
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
%   -   1.1.000     19/07/2021  Royi Avital
%       *   Updated to use modern MATLAB arguments validation.
%   -   1.0.000     20/01/2019  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments(Input)
    vX (:, :) {mustBeNumeric, mustBeVector, mustBeFinite}
    vH (:, :) {mustBeNumeric, mustBeVector, mustBeFinite}
    vG (:, :) {mustBeNumeric, mustBeVector, mustBeFinite} = 0
    convShape (1, 1) {mustBeNumeric, mustBeMember(convShape, [1, 2, 3])} = 1
    paramLambda (1, 1) {mustBeReal, mustBeNonnegative, mustBeFinite} = 0
end

arguments(Output)
    vY (:, :) {mustBeNumeric, mustBeVector, mustBeFinite}
end

CONVOLUTION_SHAPE_FULL  = 1;
CONVOLUTION_SHAPE_SAME  = 2;
CONVOLUTION_SHAPE_VALID = 3;

D_CONV_STR = dictionary(CONVOLUTION_SHAPE_FULL, 'full', CONVOLUTION_SHAPE_SAME, 'same', CONVOLUTION_SHAPE_VALID, 'valid');

vY = conv2(vX, vH, D_CONV_STR(convShape));
[vHAdj, convShapeAdj] = CalcAdjointOperator(vH, convShape);
vY = conv2(vY, vHAdj, D_CONV_STR(convShapeAdj));

if((paramLambda > 0) && any(vG))
    vZ = conv2(vX, vG, D_CONV_STR(convShape));
    [vGAdj, convShapeAdj] = CalcAdjointOperator(vG, convShape);
    vZ = conv2(vZ, vGAdj, D_CONV_STR(convShapeAdj));

    vY = vY + (paramLambda * vZ);
end


end


function [ vHAdj, convShapeAdj ] = CalcAdjointOperator( vH, convShape )

CONVOLUTION_SHAPE_FULL  = 1;
CONVOLUTION_SHAPE_SAME  = 2;
CONVOLUTION_SHAPE_VALID = 3;

switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        convShapeAdj    = CONVOLUTION_SHAPE_VALID;
        vHAdj           = flip(vH);
    case(CONVOLUTION_SHAPE_SAME)
        convShapeAdj    = CONVOLUTION_SHAPE_SAME;
        vHAdj           = flip([vH; 0]); %<! Solve issues for even length filters (Correlation has the anchor on different place)
    case(CONVOLUTION_SHAPE_VALID)
        convShapeAdj    = CONVOLUTION_SHAPE_FULL;
        vHAdj          = flip(vH);
end

end

