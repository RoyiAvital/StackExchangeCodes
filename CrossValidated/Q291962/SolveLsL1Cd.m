function [ vX, mX ] = SolveLsL1Cd( mA, vB, paramLambda, numIterations )
% ----------------------------------------------------------------------------------------------- %
%[ vX, mX ] = SolveLsL1Cd( mA, vB, lambdaFctr, numIterations )
% Solve L1 Regularized Least Squares Using Coordinate Descent (CD) Method.
% Input:
%   - mA                -   Input Matrix.
%                           The model matrix.
%                           Structure: Matrix (m X n).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vB                -   input Vector.
%                           The model known data.
%                           Structure: Vector (m X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - paramLambda       -   Parameter Lambda.
%                           The L1 Regularization parameter.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - numIterations     -   Number of Iterations.
%                           Number of iterations of the algorithm.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range {1, 2, ...}.
% Output:
%   - vX                -   Output Vector.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References
%   1.  Wikipedia CD - https://en.wikipedia.org/wiki/Coordinate_descent.
% Remarks:
%   1.  Using vanilla CD.
% Known Issues:
%   1.  A
% TODO:
%   1.  B
% Release Notes:
%   -   1.0.000     23/08/2017
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %


vANorm = sum(mA .^ 2, 1);
numElements = size(mA, 2); %<! Size of solution

vX  = pinv(mA) * vB; %<! Dealing with "Fat Matrix"

mX = zeros([size(vX, 1), numIterations]);
mX(:, 1) = vX;

for ii = 2:numIterations
    
    for jj = 1:numElements
        vA = mA(:, jj);
        colNormSqr = vANorm(jj);
        
        vExcCoord = [1:jj - 1, jj + 1:numElements];
        
        vR = vB - (mA(:, vExcCoord) * vX(vExcCoord));
        
        vBeta = vA.' * vR;
        
        
        % vX(jj) = ProxL1( vBeta / colNormSqr, paramLambda / colNormSqr );
        vX(jj) = ProxL1( vBeta, paramLambda ) / colNormSqr;
    end
    
    mX(:, ii) = vX;
    
end


end


function [ vX ] = ProxL1( vX, lambdaFactor )

% Soft Thresholding
vX = max(vX - lambdaFactor, 0) + min(vX + lambdaFactor, 0);


end

