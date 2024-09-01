function [ vX, mX ] = SolveLsL1ComplexRealCd( mA, vB, lambdaFctr, numIterations )
% ----------------------------------------------------------------------------------------------- %
%[ vX, mX ] = SolveLsL1ComplexPgm( mA, vB, lambdaFctr, numIterations )
% Solves the 0.5 * || A x - b ||_2 + \lambda || x ||_1 problem using
% Coordinate Descent Method. The model allows A, b and x to be Complex.
% This algorithm solves the problem by transforming the problem into the
% Real Domain.
% Input:
%   - mA                -   Model Matrix.
%                           The model matrix.
%                           Structure: Matrix (m X n).
%                           Type: 'Single' / 'Double' (Complex).
%                           Range: (-inf, inf).
%   - vB                -   Input Vector.
%                           The model known data.
%                           Structure: Vector (m X 1).
%                           Type: 'Single' / 'Double' (Complex).
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
%   1.  Wikipedia Coordinate Descent Method - https://en.wikipedia.org/wiki/Coordinate_descent.
% Remarks:
%   1.  Coordinate Descent is basically Steepest Descent in L1 Norm.
%   2.  This implementation, using transformation into Real Domain, does
%       Coordinate Descent in 2 coordinates (jj, jj + numElements) which
%       are tied due to the Complex Norm. Basically generalizing the
%       solution in the Complex - Real PGM.
% Known Issues:
%   1.  A
% TODO:
%   1.  B
% Release Notes:
%   -   1.0.000     07/11/2016
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

vX  = pinv(mA) * vB; %<! Dealing with "Fat Matrix"

numRows = size(vX, 1);

mB = [real(mA), -imag(mA); imag(mA), real(mA)];
vC = [real(vB); imag(vB)];

vW = [real(vX); imag(vX)];

% 0.5 * || mB * vZ - vC ||^2 + g(vZ)

vBNorm = sum(mB .* conj(mB), 1);
numElements = size(mB, 2); %<! Size of solution
numElements = numElements / 2;

mX = zeros([size(vX, 1), numIterations]);
mX(:, 1) = vX;

for ii = 2:numIterations
    
    for jj = 1:numElements
         vB = mB(:, [jj, jj + numElements]);
         colNormSqr = vBNorm(jj);
         
         vExcCoord = [1:(jj - 1), (jj + 1):numElements, (numElements + 1):(numElements + jj - 1), (numElements + jj + 1):(numElements + numElements)];
         
         vR = vC - (mB(:, vExcCoord) * vW(vExcCoord));
         vBeta = vB' * vR;
         
         % vW([jj, jj + numElements]) = ProxBlockL2(vBeta / colNormSqr, lambdaFctr / colNormSqr);
         vW([jj, jj + numElements]) = ProxBlockL2(vBeta, lambdaFctr) / colNormSqr;
    end
    
    vX = vW(1:numRows) + (1i * vW((numRows + 1):end));
    mX(:, ii) = vX;
    
end


end


function [ vX ] = ProxBlockL2( vX, lambdaFactor )

numRows = size(vX, 1) / 2;

% vY = vX(1:numRows);
% vZ = vX((numRows + 1):end);
% 
% vSqrtFactor = sqrt((vY .* vY) + (vZ .* vZ));

vSqrtFactor = repmat([sqrt((vX(1:numRows) .* vX(1:numRows)) + (vX((numRows + 1):end) .* vX((numRows + 1):end)))], [2, 1]);

vX = vX .* max((1 - (lambdaFactor ./ vSqrtFactor)), 0);


end

