function [ vX, mX ] = SolveLsL1Admm( mA, vB, lambdaFctr, numIterations )
% ----------------------------------------------------------------------------------------------- %
%[ vF ] = BayesianDft( vX, numFreqBins, varX, varN, numIterations )
% High Resolution DFT using Bayesian Estimation of the DFT coefficients.
% Input:
%   - vX                -   Input Vector.
%                           Structure: Vector (Column Vector).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - numFreqBins       -   Number of Frequency Bins.
%                           The number of Frequency Bins in the Frequency
%                           Domain.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, ...}.
%   - varX              -   Variance of Signal Model.
%                           The variance of each peak in the frequency
%                           domain assuming Normal Distribution.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - varN              -   Variance of Noise.
%                           The variance of the noise.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - numIterations     -   Number of Iterations.
%                           Number of iterations of the algorithm.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range {1, 2, ...}.
% Output:
%   - vF                -   High Resolution DFT.
%                           High Resolution DFT generated according to the
%                           Bayesian Model.
%                           Structure: Vector (Column Vector).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References
%   1.  C
% Remarks:
%   1.  T
% TODO:
%   1.  Use Levinson's Recursion to solve `vB = ((lambdaFctr * mI) + mF * mQ * mF') \ vX;`.
%   2.  Add "Stopping Condition".
% Release Notes:
%   -   1.0.000     07/11/2016
%       *   First realease version.
% ----------------------------------------------------------------------------------------------- %

rhoFctr = 1.0;

mAA = mA' * mA;
mI  = eye(size(mAA, 1));
mAAInv = inv(mAA + (rhoFctr * mI));
vAb = mA' * vB;
% vX  = mAA \ vAb;
vX  = pinv(mA) * vB; %<! Dealing with "Fat Matrix"

mX = zeros([size(vX, 1), numIterations]);
mX(:, 1) = vX;
vY = zeros([size(vX, 1), 1]);
vZ = zeros([size(vX, 1), 1]);

for ii = 2:numIterations
    vX = mAAInv * (vAb + (rhoFctr * vY) - vZ);
    vY = ProxL1((vZ / rhoFctr) + vX, (lambdaFctr / rhoFctr));
    vZ = vZ + (rhoFctr * (vX - vY));
    
    mX(:, ii) = vX;
end


end


function [ vX ] = ProxL1( vX, lambdaFactor )

% Soft Thresholding - Complex Domain -> Keep Phase, Soft Threshold the
% Modulus

% vXAbs   = abs(vX);
% vXPhase = angle(vX);
% 
% vX = max(vXAbs - lambdaFactor, 0) .* exp(1i * vXPhase);

vXAbs = abs(vX);

vX              = (vX ./ vXAbs) .* max((vXAbs - lambdaFactor), 0);
vX(vXAbs == 0)  = 0;

end

