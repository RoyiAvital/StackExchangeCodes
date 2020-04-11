function [ mX ] = ProjectPosSemiDefinite( mY, numIterations, stopTol )
% ----------------------------------------------------------------------------------------------- %
% [ mX ] = ProjectPosSemiDefinite( mY, numIterations, stopTol )
%   Solves \arg \min_{X} 0.5 || X - Y ||, s.t. X \in \mathcal{S}_{+}^{n} by
%   Iterative Projection onto the set of Symmetric Matrices and PSD
%   Matrices.
% Input:
%   - mY            -   Input Matrix.
%                       Input model matrix.
%                       Structure: Matrix (m x n).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - numIterations -   Number of Iterations.
%                       Sets the number of iterations for the algorithm to
%                       run.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range {1, 2, ...}.
%   - stopTol       -   Stopping Condition Tolerance.
%                       Sets the stopping threshold for the L Inf (Maximum
%                       Absolute Value) of the change between 2 iterations
%                       of the algorithm.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range [0, inf).
% Output:
%   - mX            -   Solution Matrix.
%                       A PSD Symmetric matrix which is the closest in
%                       Frobenius Norm sense to 'mY'.
%                       Structure: Matrix (m x n).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  A
% Remarks:
%   1.  Since both the Symmetric Matrices Set and the PSD Matrices Set are
%       sub spaces the iterative (POCS) projection onto each is the
%       orthogonal projection onto the intersection. See https://math.stackexchange.com/questions/1492095.
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     12/04/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

vectorMode = OFF;

if(size(mY, 2) == 1)
    numRows     = sqrt(size(mY, 1));
    vectorMode  = ON;
    mX          = reshape(mY, numRows, numRows);
else
    numRows = size(mY, 1);
    mX      = mY;
end

mQ      = zeros(numRows, numRows);
mD      = zeros(numRows, numRows);
mXPrev  = mX;

for ii = 1:numIterations
    % Projection onto Symmetric Matrices Set
    mX(:) = (mX.' + mX) / 2;
    
    % Projection onto PSD Matrices Set
    [mQ(:), mD(:)] = eig(mX);
    mD(:) = max(mD, 0);
    mX(:) = mQ * mD * mQ.';
    
    if(max(abs(mX(:) - mXPrev(:))) <= stopTol)
        break;
    end
    
    mXPrev(:) = mX;
    
end

if(vectorMode == ON)
    mX = mX(:);
end


end

