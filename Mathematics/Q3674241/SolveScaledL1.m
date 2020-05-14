function [ valA ] = SolveScaledL1( vX, vY )
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

valEps      = 1e-6;

numElements = size(vX, 1);
hF          = @(valA) sign(valA * vX - vY).' * vX;

vA = vY ./ vX;

for ii = 1:numElements
    valMinusEps = hF(vA(ii) - valEps);
    valPlusEps = hF(vA(ii) + valEps);
    
    if(valMinusEps * valPlusEps < 0)
        break;
    end
end

valA = vA(ii);


end

