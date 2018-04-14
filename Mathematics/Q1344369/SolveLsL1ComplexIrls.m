function [ vX, mX ] = SolveLsL1ComplexIrls( mA, vB, lambdaFctr, numIterations )
% ----------------------------------------------------------------------------------------------- %
%[ vX, mX ] = SolveLsL1ComplexIrls( mA, vB, lambdaFctr, numIterations )
% Solves the 0.5 * || A x - b ||_2 + \lambda || x ||_1 problem using Fixed
% Point Iteration Method. The model allows A, b and x to be Complex.
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
%   1.  Wikipedia Fixed Point Iteration Method - https://en.wikipedia.org/wiki/Fixed-point_iteration.
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

paramEps = 1e-9;

% vX  = mAA \ vAb;
vX  = pinv(mA) * vB; %<! Dealing with "Fat Matrix"

mAA = mA' * mA;
vAb = mA' * vB;
% mI = eye(size(mA, 1));

mX = zeros([size(vX, 1), numIterations]);
mX(:, 1) = vX;

for ii = 2:numIterations
    vX = (mAA + (lambdaFctr * diag(1./ (abs(vX) + paramEps)))) \ vAb;
    
    % Doesn't work (Bad conditioning)
    % mX = diag(abs(vX));
    % vX = (mX * mA') * (((mA * mX * mA.') + (lambdaFctr * mI)) \ vB);
    
    mX(:, ii) = vX;
end


end

