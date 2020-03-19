function [ vX ] = OrthogonalProjectionOntoConvexSets( cProjFun, vY, numIterations, stopThr )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = OrthogonalProjectionOntoConvexSets( cProjFun, vY, numIterations, stopThr )
%   Solves \arg \min_{x} 0.5 || x - y ||, s.t. x \in \bigcap {C}_{i} using
%   Dykstra's Projection Algorithm.
% Input:
%   - mA            -   Model Matrix.
%                       Input model matrix.
%                       Structure: Matrix (m x n).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - vX            -   Solution Vector.
%                       The solution to the optimization problem..
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

mZ = zeros(numElements, numSets);
mU = zeros(numElements, numSets);
vU = vY;
vX = vU;

for ii = 1:numIterations
    for jj = 1:numSets
        mU(:, jj) = cProjFun{jj}( vU + mZ(:, jj) );
        mZ(:, jj) = vU + mZ(:, jj) - mU(:, jj);
        
        vU(:) = mU(:, jj);
    end
    
    % To calculate the difference from the previous iteration.
    stopCond = max(abs(vX - vU)) < stopThr;
    
    vX(:) = vU;
    
    if(stopCond)
        break;
    end
end


end

