function [ vX, mX ] = SolveLsL1ComplexRealPgm( mA, vB, lambdaFctr, numIterations )
% ----------------------------------------------------------------------------------------------- %
%[ vX, mX ] = SolveLsL1ComplexPgm( mA, vB, lambdaFctr, numIterations )
% Solves the 0.5 * || A x - b ||_2 + \lambda || x ||_1 problem using
% Proximal Gradient Method. The model allows A, b and x to be Complex. This
% algorithm solves the problem by transforming teh problem into the Real
% Domain.
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
%   1.  Wikipedia Proximal Gradient Method - https://en.wikipedia.org/wiki/Proximal_gradient_method.
% Remarks:
%   1.  A
% Known Issues:
%   1.  A
% TODO:
%   1.  B
% Release Notes:
%   -   1.0.000     07/11/2016
%       *   First realease version.
% ----------------------------------------------------------------------------------------------- %

% vX  = mAA \ vAb;
vX  = pinv(mA) * vB; %<! Dealing with "Fat Matrix"

numRows = size(vX, 1);

mB = [real(mA), -imag(mA); imag(mA), real(mA)];
vC = [real(vB); imag(vB)];

vW = [real(vX); imag(vX)];

% 0.5 * || mB * vZ - vC ||^2 + g(vZ)

mBB = mB.' * mB;
vBC = mB.' * vC;

paramAlphaBase = 2 / (1.05 * (norm(mA, 2) ^ 2));

mX = zeros([numRows, numIterations]);
mX(:, 1) = vW(1:numRows) + (1i * vW((numRows + 1):end));

for ii = 2:numIterations
    vWGrad      = (mBB * vW) - vBC;
    
    paramAlpha  = paramAlphaBase / sqrt(ii - 1);
    vW          = ProxL1(vW - (paramAlpha * vWGrad), paramAlpha * lambdaFctr);
    
    vX = vW(1:numRows) + (1i * vW((numRows + 1):end));
    mX(:, ii) = vX;
end


end


function [ vX ] = ProxL1( vX, lambdaFactor )

numRows = size(vX, 1) / 2;

% vY = vX(1:numRows);
% vZ = vX((numRows + 1):end);
% 
% vSqrtFactor = sqrt((vY .* vY) + (vZ .* vZ));

vSqrtFactor = repmat([sqrt((vX(1:numRows) .* vX(1:numRows)) + (vX((numRows + 1):end) .* vX((numRows + 1):end)))], [2, 1]);

vX = vX .* max((1 - (lambdaFactor ./ vSqrtFactor)), 0);


end

