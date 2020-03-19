function [ vX, mX ] = SolveLsUnitSimplexProjectedGd( mA, vB, vX, numIterations, stopTol, operationMode )
% ----------------------------------------------------------------------------------------------- %
% [ vX, mX ] = SolveLsUnitSimplexProjectedGd( mA, vB, vX, numIterations, stopTol, operationMode )
%   Solves \arg \min_{x} 0.5 || A x - b ||, s.t. 0 <= x, sum(x) = 1 using
%   Projected Gradient Descent Method.
% Input:
%   - mA            -   Model Matrix.
%                       Input model matrix.
%                       Structure: Matrix (m x n).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vB            -   Measurements Vector.
%                       Given data vector.
%                       Structure: Vector (n x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - numIterations -   Number of Iterations.
%                       Sets the number of iterations for the algorithm to
%                       run.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range {1, 2, ...}.
%   - stopTol       -   Stopping Condition Tolerance.
%                       Sets the stopping threshold for the L Inf (Maximum
%                       Absolute Value) of the change between 2 iterations
%                       of the algorithm.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range [0, inf).
%   - operationMode -   Operation Mode.
%                       Sets whether to use Alternating Projections or
%                       Durect Projection for the projection onto the Unit
%                       Simplex.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range {1, 2}.
% Output:
%   - vX            -   Solution Vector.
%                       The solution to the optimization problem..
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mX            -   Solution Path Matrix.
%                       Matrix which each of its columns embeds the
%                       solution of the i-th step.
%                       Structure: Matrix (m x numIterations).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  A
% Remarks:
%   1.  If one sets 'stopTol = 0' then the algorithm will run the given
%       number of iterations.
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

OPERATION_MODE_ALTERNATING_PROJECTIONS  = 1;
OPERATION_MODE_DIRECT_PROJECTION        = 2;

numRows = size(mA, 1);
numCols = size(mA, 2);

numAltIterations    = ceil(numCols / 1);
stopThr             = 1e-5;
ballRadius          = 1;

switch(operationMode)
    case(OPERATION_MODE_ALTERNATING_PROJECTIONS)
        hProjFunction = @(vX) AlternatingProjection(vX, numAltIterations, numCols);
    case(OPERATION_MODE_DIRECT_PROJECTION)
        hProjFunction = @(vX) ProjectSimplex(vX, ballRadius, stopThr);
end

mAA = mA.' * mA;
vAb = mA.' * vB;

% Lipschitz Constant
% stepSize = 1 / norm(mA, 2);
% stepSize = 1 / (2 * (norm(mA, 2) ^ 2));
stepSize = 2 / sum(mA(:) .^ 2); %<! Faster to calculate, conservative (Hence slower)

mX = zeros(numCols, numIterations);
mX(:, 1) = vX;

for ii = 2:numIterations
    vG = mAA * vX - vAb;
    vX(:) = vX - (stepSize * vG);
    
    vX(:) = hProjFunction(vX);
    
    mX(:, ii) = vX;
    
    if(max(abs(vX - mX(:, ii - 1))) <= stopTol)
        break;
    end
end


end


function [ vX ] = AlternatingProjection( vX, numIterations, numCols )

for ii = 1:numIterations
    
    % Projection onto Non Negative Orthant
    vX(:) = max(vX, 0);
    % Projection onto Sum of 1
    vX(:) = vX - ((sum(vX) - 1) / numCols);
    
end


end

