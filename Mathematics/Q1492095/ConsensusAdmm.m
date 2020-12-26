function [ vX ] = ConsensusAdmm( cProxFun, numElements, paramRho, numIterations, stopThr )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ConsensusAdmm( cProxFun, numElements, paramRho, numIterations, stopThr )
%   Solves the Consensus ADMM problem given the set of Proximal Operators.
% Input:
%   - cProxFun      -   Set of Proximal Operators.
%                       Each cell elemnts is the i-th proximal operator.
%                       Each function handler input is a (vV, paramLambda).
%                       The implementation is for the prox in its lambda
%                       form: 0.5 * || x - v ||^2 + paramLamba * g(x).
%                       Structure: Cell Array (numSets x 1).
%                       Type: Function Handler.
%                       Range: NA.
% Output:
%   - vX            -   Solution Vector.
%                       The solution to the optimization problem..
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  ELE 522 - Large Scale Optimization for Data Science - Alternating Direction Method of Multipliers.
% Remarks:
%   1.  Pay attention that the references are using (1 / paramLambda) form
%       of the Prox. Hence the adaptation in the code.
%   2.  This is ultra efficient way to solve the Orthogonal Projection
%       Problem in case having projection onto sets which the intersection
%       is the objective set. In that case each Prox Function is a
%       projection function.
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     25/03/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numSets     = size(cProxFun, 1);
% numElements = size(vX, 1);

paramRhoInv = 1 / paramRho;

mX      = zeros(numElements, numSets);
vZ      = mean(mX, 2);
mLambda = zeros(numElements, numSets);

for ii = 1:numIterations
    for jj = 1:numSets
        mX(:, jj) = cProxFun{jj}(vZ - (paramRhoInv * mLambda(:, jj)), paramRhoInv);
    end
    
    vZ = mean(mX + (paramRhoInv * mLambda), 2);
    
    for jj = 1:numSets
        mLambda(:, jj) = mLambda(:, jj) + (paramRho * (mX(:, jj) - vZ)); %<! Can be done mLambda = mLambda + (paramRho * (mX - vZ));
    end
    
    % To calculate the difference from the previous iteration.
    stopCond = max(abs(mX - vZ), [], 'all') < stopThr;
    
    if(stopCond)
        break;
    end
end

vX = mean(mX, 2);


end

