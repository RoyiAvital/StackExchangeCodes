function [ mLU ] = GenerateTriangleExtractorMatrix( numRows, triangleFlag, diagFlag )
% ----------------------------------------------------------------------------------------------- %
% [ mLU ] = GenerateTriangleExtractorMatrix( numRows, triangleFlag, diagFlag )
% Generates a sparse Matrix which extracts the Lower / Upper Triangle of a
% matrix from its vectorized form. So 'mLU * mX(:)' extracts a vector which
% are the elements of a triangle of the square matrix 'mX'.
% Input:
%   - numRows           -   Number of Rows.
%                           Number of rows of the square matrix 'mX'.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {2, 3, ...}.
%   - triangleFlag      -   Triangle Flag.
%                           Sets whether to extract the lower or the upper
%                           triangle.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2}.
%   - diagFlag          -   Diagonal Flag.
%                           Sets whether to extract the main diagonal or
%                           not.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2}.
% Output:
%   - mLU               -   Output Sparse Matrix.
%                           The extraction operator.
%                           Structure: Matrix (Sparse).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  Most Efficient Way to Construct the Matrices to Extract the Lower and Upper Triangle from a Vectorized Matrix - https://www.mathworks.com/matlabcentral/answers/519408.
% Remarks:
%   1.  Generally for lower triangle 'mLU * mX(:)' should be equivalent of
%       'mX(logical(tril(mX, -1))))' when diagonal excluded.
%   2.  For upper triangular the loop goes backwards in order to keep data
%       in column wise form as expected by the vectorization operator.
%   3.  The matrix 'mX' is assumed to be square matrix.
% TODO:
%   1.  Extend the code to support non square matrices.
%   Release Notes:
%   -   1.0.000     21/04/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

EXTRACT_LOWER_TRIANGLE = 1;
EXTRACT_UPPER_TRIANGLE = 2;

INCLUDE_DIAGONAL = 1;
EXCLUDE_DIAGONAL = 2;

switch(diagFlag)
    case(INCLUDE_DIAGONAL)
        numElements = 0.5 * numRows * (numRows + 1);
        diagIdx = 0;
    case(EXCLUDE_DIAGONAL)
        numElements = 0.5 * (numRows - 1) * numRows;
        diagIdx = 1;
end

vJ = zeros(numElements, 1); %<! The column Index of the Sparse Matrix

if(triangleFlag == EXTRACT_LOWER_TRIANGLE)
    % Lower Triangle
    elmntIdx = 0;
    for jj = 1:numRows
        for ii = (jj + diagIdx):numRows
            elmntIdx = elmntIdx + 1;
            vJ(elmntIdx) = ((jj - 1) * numRows) + ii;
        end
    end
elseif(triangleFlag == EXTRACT_UPPER_TRIANGLE)
    % Upper Triangle
    % Going from the end to start in order to be effiecent ('ii' can start
    % from 'jj - diagIdx') and keep the column wise form.
    elmntIdx = numElements + 1;
    for jj = numRows:-1:1
        for ii = (jj - diagIdx):-1:1
            elmntIdx = elmntIdx - 1;
            vJ(elmntIdx) = ((jj - 1) * numRows) + ii;
        end
    end
end

mLU = sparse(1:numElements, vJ, 1, numElements, numRows * numRows, numElements);


end

