function [ vX ] = AlternatingProjectionOntoConvexSets( cProjFun, vY, numIterations, stopThr )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = AlternatingProjectionOntoConvexSets( cProjFun, vY, numIterations, stopThr )
%   Solves x \in \bigcap {C}_{i} using Alternating Projections algorithm.
%   In case all sets to be projected at are sub spaces this matches the
%   Dykstra's Projection Algorithm. Otherwise it only guarantees to
%   converge to the intersection of the sets in case it is not empty.
% Input:
% Input:
%   - cProjFun      -   Array of Projection Functions.
%                       Cell array of anonymouse functions which each is a
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
%   1.  Dykstra's Projection Algorithm (Wikipedia) - https://en.wikipedia.org/wiki/Dykstra%27s_projection_algorithm.
%   2.  Ryan J. Tibshirani - Dykstra’s Algorithm, ADMM, and Coordinate Descent: Connections, Insights, and Extensions.
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     19/03/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numSets = size(cProjFun, 1);
numElements = size(vY, 1);

vX = vY;

for ii = 1:numIterations
    vY(:) = vX; %<! Using it as a buffer for the previous iteration
    for jj = 1:numSets
        vX(:) = cProjFun{jj}(vX);
    end
    
    % To calculate the difference from the previous iteration.
    stopCond = max(abs(vX - vY)) < stopThr;
    
    if(stopCond)
        break;
    end
end


end

