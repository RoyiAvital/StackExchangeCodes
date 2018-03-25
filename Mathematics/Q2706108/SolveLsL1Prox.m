function [ vX, mX ] = SolveLsL1Prox( mA, vB, paramLambda, numIterations )
% ----------------------------------------------------------------------------------------------- %
%[ vX, mX ] = SolveLsL1Prox( mA, vB, lambdaFctr, numIterations )
% Solve L1 Regularized Least Squares Using Proximal Gradient (PGM) Method.
% Input:
%   - mA                -   Input Matirx.
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
%   1.  Wikipedia PGM - https://en.wikipedia.org/wiki/Proximal_gradient_method.
% Remarks:
%   1.  Using vanilla PGM.
% Known Issues:
%   1.  A
% TODO:
%   1.  B
% Release Notes:
%   -   1.0.000     23/08/2017
%       *   First realease version.
% ----------------------------------------------------------------------------------------------- %

mAA = mA.' * mA;
vAb = mA.' * vB;
vX  = pinv(mA) * vB; %<! Dealing with "Fat Matrix"

stepSize = 1 / (2 * (norm(mA, 2) ^ 2));
% stepSize = 1 / sum(mA(:) .^ 2); %<! Faster to calculate, conservative (Hence slower)

mX = zeros([size(vX, 1), numIterations]);
vX = mX(:, 1);

for ii = 2:numIterations
    
    vG = (mAA * vX) - vAb;
    vX = ProxL1(vX - (stepSize * vG), stepSize * paramLambda);
    vX = max(vX, 0); %<! Project onto R+
    
    mX(:, ii) = vX;
    
end


end


function [ vX ] = ProxL1( vX, lambdaFactor )

% Soft Thresholding
vX = max(vX - lambdaFactor, 0) + min(vX + lambdaFactor, 0);


end

