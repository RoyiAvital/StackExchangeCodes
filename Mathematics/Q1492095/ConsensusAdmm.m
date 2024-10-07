function [ vX ] = ConsensusAdmm( cProxFun, numElements, paramRho, numIterations, stopThr )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ConsensusAdmm( cProxFun, numElements, paramRho, numIterations, stopThr )
%   Solves the Consensus ADMM problem given the set of Proximal Operators.
% Input:
%   - cProxFun      -   Set of Proximal Operators.
%                       Each cell elements is the i-th proximal operator.
%                       Each function handler input is a (vV, paramLambda).
%                       The implementation is for the prox in its lambda
%                       form: 0.5 * || x - v ||^2 + paramLambda * g(x).
%                       Structure: Cell Array (numSets x 1).
%                       Type: Function Handler.
%                       Range: NA.
%   - numElements   -   Number of Elements.
%                       The number of elements of the data vector.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ...}.
%   - paramRho      -   Parameter Rho.
%                       The value of the Augmented Lagrangian Multiplier
%                       (ALM).
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
%   - numIterations -   Number of Iterations.
%                       Sets the number of iterations of the algorithm.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ...}.
%   - stopThr       -   Stopping Threshold.
%                       Sets the threshold between consecutive iterations
%                       for stopping the algorithm.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
% Output:
%   - vX            -   Solution Vector.
%                       The solution to the optimization problem.
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  ELE 522 - Large Scale Optimization for Data Science - Alternating Direction Method of Multipliers.
% Remarks:
%   1.  Pay attention that the references are using (1 / paramLambda) form
%       of the Prox which is the standard form of the ADMM. Hence, in the
%       code the parameter rho is inverted. The form used in the ADMM is:
%       (rho / 2) * || x - v ||^2 + g(x).
%   2.  This is ultra efficient way to solve the Orthogonal Projection
%       Problem in case having projection onto sets which the intersection
%       is the objective set. In that case each Prox Function is a
%       projection function.
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.001     28/12/2020  Royi Avital
%       *   Updated to the scaled form of the ADMM.
%   -   1.0.000     25/03/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numSets     = size(cProxFun, 1);
% numElements = size(vX, 1);

% The prox is usually given by: \arg \min_x 0.5 * || x - v ||^2 + paramLambda * g(x).
% The ADMM form is: \arg \min_x (rho / 2) * || x - v ||^2 + g(x). 
% Which means paramLamba = 1 / paramRho.
paramRhoInv = 1 / paramRho; %<! Basically paramLambda above.

mX      = zeros(numElements, numSets);
vZ      = mean(mX, 2);
mLambda = zeros(numElements, numSets);

for ii = 1:numIterations
    for jj = 1:numSets
        mX(:, jj) = cProxFun{jj}(vZ - mLambda(:, jj), paramRhoInv);
    end
    
    vZ = mean(mX + mLambda, 2);
    
    for jj = 1:numSets
        mLambda(:, jj) = mLambda(:, jj) + mX(:, jj) - vZ; %<! Can be done mLambda = mLambda + (paramRho * (mX - vZ));
    end
    
    % To calculate the difference from the previous iteration.
    stopCond = max(abs(mX - vZ), [], 'all') < stopThr;
    
    if(stopCond)
        break;
    end
end

vX = mean(mX, 2);


end

