function [ vX ] = OrthogonalProjectionOntoConvexSetsAdmm( cProjFun, vY, numIterations, stopThr )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = OrthogonalProjectionOntoConvexSetsAdmm( cProjFun, vY, numIterations, stopThr )
%   Solves \arg \min_{x} 0.5 || x - y ||, s.t. x \in \bigcap {C}_{i} using
%   ADMM with the consensus optimization trick.
% Input:
%   - cProjFun      -   Array of Projection Functions.
%                       Cell array of anonymous functions which each is a
%                       projection into a sub space.
%                       Structure: Cell Array.
%                       Type: NA.
%                       Range: NA.
%   - vY            -   Input Vector.
%                       Input vector to be projected.
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
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
%   1.  Uses the consensus trick with the function being the least squares
%       distance from the input and the rest of the projection functions.
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     26/12/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numSets     = size(cProjFun, 1);
paramRho    = 0.1;
numCols     = size(vY, 1);

cProxFun    = cell(numSets + 1, 1);
cProxFun{1} = @(vV, paramRho) ((paramRho * vV) + vY) / (1 + paramRho); %<! Prox of (rho / 2) * || x - y ||_{2}^{2} + (1 / 2) * || x - v ||

for ii = 2:numSets + 1
    cProxFun{ii} = @(vV, paramRho) cProjFun{ii - 1}(vV);
end

vX = ConsensusAdmm(cProxFun, numCols, paramRho, numIterations, stopThr);


end

