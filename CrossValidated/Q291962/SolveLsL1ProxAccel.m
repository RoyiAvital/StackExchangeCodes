function [ vX, mX ] = SolveLsL1ProxAccel( mA, vB, paramLambda, numIterations )
% ----------------------------------------------------------------------------------------------- %
%[ vX, mX ] = SolveLsL1ProxAccel( mA, vB, paramLambda, numIterations )
% Solve L1 Regularized Least Squares Using Accelerated Proximal Gradient (PGM) Method.
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
%   1.  Wikipedia PGM - https://en.wikipedia.org/wiki/Proximal_gradient_method.
%   2.  Wikipedia Fast Gradient Methods - https://en.wikipedia.org/wiki/Gradient_descent#Fast_gradient_methods.
% Remarks:
%   1.  Using Smooth / Aggressive FISTA step size (The smooth is the
%       original step size in the FISTA article).
% Known Issues:
%   1.  A
% TODO:
%   1.  B
% Release Notes:
%   -   1.0.000     23/08/2017
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FISTA_STEP__METHOD_SMOOTH       = 1; %<! Monotonic, Slower
FISTA_STEP__METHOD_AGGRESSIVE   = 2; %<! Non Monotonic, Faster

mAA = mA.' * mA;
vAb = mA.' * vB;
vX  = pinv(mA) * vB; %<! Dealing with "Fat Matrix"

stepSize = 1 / (2 * (norm(mA, 2) ^ 2));
% stepSize = 1 / sum(mA(:) .^ 2); %<! Faster to calculate, conservative (Hence slower)
fistaStepMode = FISTA_STEP__METHOD_AGGRESSIVE;

mX = zeros([size(vX, 1), (numIterations + 1)]);
mX(:, 1) = vX;

vY = vX;
tK = 1;

for ii = 1:numIterations
    
    vYGrad      = (mAA * vY) - vAb;
    
    vXPrev      = vX;
    vX          = ProxL1(vY - (stepSize * vYGrad), stepSize * paramLambda);
    
    switch(fistaStepMode)
        case(FISTA_STEP__METHOD_SMOOTH)
            tKPrev          = tK;
            tK              = (1 + sqrt(1 + (4 * tKPrev))) / 2;
            fistaStepSize   = (tKPrev - 1) / tK;
        case(FISTA_STEP__METHOD_AGGRESSIVE)
            fistaStepSize = (ii - 1) / (ii + 2);
    end
    
    vY = vX + (fistaStepSize * (vX - vXPrev));
    
    mX(:, (ii + 1)) = vX;
    
end


end


function [ vX ] = ProxL1( vX, lambdaFactor )

% Soft Thresholding
vX = max(vX - lambdaFactor, 0) + min(vX + lambdaFactor, 0);

end

