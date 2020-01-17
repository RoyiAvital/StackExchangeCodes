function [ mH ] = GenerateToeplitzConvMatrix( vH, numElementsData, convShape )
% ----------------------------------------------------------------------------------------------- %
%[ mH ] = GenerateToeplitzConvMatrix( vH, numElementsData, convShape )
% Generates matrix 'mH' such that mH * vX = conv(vX, vH, 'convShape').
% Input:
%   - vH                -   Input Vector.
%                           The coeeficients of the filter.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - numElementsData   -   Number of Elements of the Data.
%                           Number of elements in the vector to convolve.
%                           In the equation above it is 'length(vX)'.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, ...}.
%   - convShape         -   Convolution Shape.
%                           Sets the convolution type. Using same
%                           definitions as in MATLAB 'conv()' function.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3}.
% Output:
%   - mH                -   Toeplitz Convolution Matrix.
%                           Structure: Matrix.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References
%   1.  A
% Remarks:
%   1.  B
% Known Issues:
%   1.  C
% TODO:
%   1.  Look for direct methods to calculate the matrix of the convolution
%       shape same.
% Release Notes:
%   -   1.0.000     11/01/2020  Royi Avital
%       *   First realease version.
% ----------------------------------------------------------------------------------------------- %

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;

numTaps = length(vH);

switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        vR = [vH(1); zeros(numElementsData - 1, 1)];
        vC = [vH(:); zeros(numElementsData - 1, 1)];
    case(CONVOLUTION_SHAPE_SAME)
        vR = [vH(1); zeros(numElementsData - 1, 1)];
        vC = [vH(:); zeros(numElementsData - 1, 1)];
    case(CONVOLUTION_SHAPE_VALID)
        vR = [flip(vH(:), 1); zeros(numElementsData - numTaps, 1)];
        vC = [vH(end); zeros(numElementsData - numTaps, 1)];
end

mH = toeplitz(vC, vR);

if(convShape == CONVOLUTION_SHAPE_SAME)
    firstIdx = ceil((numTaps + 1)/ 2);
    mH = mH(firstIdx:(firstIdx + numElementsData - 1), :);
end


end

