function [ vX, mX ] = SolveLsL1ComplexRealSubGrad( mA, vB, lambdaFctr, numIterations )
% ----------------------------------------------------------------------------------------------- %
%[ vX, mX ] = SolveLsL1ComplexRealSubGrad( mA, vB, lambdaFctr, numIterations )
% Solves the 0.5 * || A x - b ||_2 + \lambda || x ||_1 problem using Sub
% Gradient Method. The model allows A, b and x to be Complex. This
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
%   1.  Wikipedia Sub Gradient Method - https://en.wikipedia.org/wiki/Subgradient_method.
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

% paramAlphaBase = 2 / (1.05 * (norm(mA, 2) ^ 2));

mB = real(mA);
mC = imag(mA);
vC = real(vB);
vD = imag(vB);
vY = real(vX);
vZ = imag(vX);

mD = [mB, -mC; mC, mB];
vE = [vC; vD];
vW = [vY; vZ];

vDE = mD.' * vE;
mDD = mD.' * mD;

paramAlphaBase = 2 / (1.05 * (norm(mD, 2) ^ 2)); 

vCmplxNrm = zeros([(numRows * 2), 1]);

mX = zeros([numRows, numIterations]);
mX(:, 1) = vX;

for ii = 2:numIterations
    vCmplxNrm(:)    = repmat(sqrt(sum(reshape((vW .* vW), [numRows, 2]), 2)), [2, 1]);
    vWGrad          = (mDD * vW) - vDE + (lambdaFctr * (vW ./ vCmplxNrm));
    
    paramAlpha  = paramAlphaBase / sqrt(ii);
    vW          = vW - (paramAlpha * vWGrad);
    
    vX = vW(1:numRows) + (1i * vW((numRows + 1):end));
    mX(:, ii) = vX;
end


end

