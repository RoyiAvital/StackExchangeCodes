function [ mLU ] = GenerateSymmetricConstraintMatrix( numRows )
% ----------------------------------------------------------------------------------------------- %
% [ mLU ] = GenerateSymmetricConstraintMatrix( numRows )
% Generates a sparse Matrix which enforces symmetric constraint on a matrix
% in its vector form. For any symmetric matrix 'mX' the following holds:
% 'mLU * mX(:) = 0'. So by adding the constairnt to problem it enforces the
% solution to be a symmetric matrix.
% Input:
%   - numRows           -   Number of Rows.
%                           Number of rows of the square matrix 'mX'.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {2, 3, ...}.
% Output:
%   - mLU               -   Output Sparse Matrix.
%                           The constraint in a matrix form.
%                           Structure: Matrix (Sparse).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  A
% Remarks:
%   1.  The matrix 'mX' is assumed to be square matrix.
% TODO:
%   1.  Extend the code to support non square matrices.
%   Release Notes:
%   -   1.0.000     21/04/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

numElements = (numRows - 1) * numRows; %<! 2 * 0.5 * (numRows - 1) * numRows;

vJ = zeros(numElements, 1); %<! The column Index of the Sparse Matrix
vV = zeros(numElements, 1); %<! The values of the Sparse Matrix

elmntIdx = 0;
for jj = 1:numRows
    for ii = (jj + 1):numRows
        elmntIdx = elmntIdx + 1;
        vJ(elmntIdx) = ((jj - 1) * numRows) + ii;
        vV(elmntIdx) = 1;
        elmntIdx = elmntIdx + 1;
        vJ(elmntIdx) = ((ii - 1) * numRows) + jj;
        vV(elmntIdx) = -1;
    end
end

mLU = sparse(reshape(repmat(1:(numElements / 2), 2, 1), numElements, 1), vJ, vV, numElements, numRows * numRows, numElements);


end

