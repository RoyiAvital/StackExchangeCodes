function [ mD ] = CreateGradientOperator( numRows, numCols )
% ----------------------------------------------------------------------------------------------- %
% [ mD ] = CreateGradientOperator( numRows, numCols )
% Generates a Convolution Matrix for the 2D Gradient of the form [-1, 1].
% Input:
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
% Output:
%   - mD                -   Convolution Matrix.
%                           The output convolution matrix. The product of
%                           the matrix 'mD' and and image 'mI' in its
%                           column stack form ('mD * mI(:)') is equivalent
%                           to the convolution of 'mI' with the kernel
%                           using the valid convolution shape
%                           ('conv2(mI, mH, 'valid')').
%                           Structure: Matrix (Sparse).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  See my notes - 'Matrix Form of Image Gradient.md'.
% Remarks:
%   1.  The function basically calculates the Convolution Matrix for the
%       kernel [-1, 1] with operation mode of Valid Convolution. See
%       'CreateGradientOperatorUnitTest()' for the exact operation.
%   2.  The matrix 'mT' is the template for creating Vertical / Horizontal
%       derivative operator.
% TODO:
%   1.  
%   Release Notes:
%   -   1.0.000     25/03/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

% Vertical Operator - T(numRows)
mT  = spdiags([ones(numRows - 1, 1), -ones(numRows - 1, 1)], [0, 1], numRows - 1, numRows);
mDv = kron(eye(numCols), mT);

% Vertical Operator - T(numCols)
mT  = spdiags([ones(numCols, 1), -ones(numCols, 1)], [0, 1], numCols - 1, numCols);
mDh = kron(mT, eye(numRows));

mD = [mDv; mDh];


end

