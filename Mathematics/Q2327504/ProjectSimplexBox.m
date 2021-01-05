function [ vX ] = ProjectSimplexBox( vY, ballRadius, maxVal, stopThr )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectSimplex( vY, ballRadius, stopThr )
%   Solving the Orthogonal Projection Problem of the input vector onto the
%   Simplex Ball with upper bound constraint using Dual Function.
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (numElements x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - ballRadius    -   Ball Radius.
%                       Sets the Radius of the Simplex Ball. For Unit
%                       Simplex set to 1.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
%   - maxVal        -   Maximum Value.
%                       Sets the upper bound of the values of the
%                       solutions.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: [ballRadius / numElements, inf).
%   - stopThr       -   Stopping Threshold.
%                       Sets the threshold of the 1D Optimization. The
%                       absolute value of the Objective Function will be
%                       below the threshold.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
% Output:
%   - vX            -   Output Vector.
%                       The projection of the Input Vector onto the Simplex
%                       Ball.
%                       Structure: Vector (numElements x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  Orthogonal Projection onto a Variation of the Unit Simplex (https://math.stackexchange.com/questions/3972913).
% Remarks:
%   1.  The objective function isn't smooth (Piece wise linear). Hence it
%       is better to utilize methods for continuous functions.
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     05/01/2020  Royi Avital     RoyiAvital@yahoo.com
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numElements = size(vY, 1);

if(maxVal < (ballRadius / numElements))
    error(['The value of `maxVal` must be not less than  `ballRadius / size(vX, 1)`']);
end

if((abs(sum(vY) - ballRadius) < (0.1 * stopThr)) && all(vY >= 0) && all(vY <= maxVal))
    % The input is already within the set.
    vX = vY;
    return;
end

if(abs((numElements * maxVal) - ballRadius) < (0.1 * stopThr))
    % The input is already within the set.
    vX = maxVal * ones(numElements, 1);
    return;
end

% The objective functions which its root (The 'paramMu' which makes it
% vanish) is the solution
hObjFun = @(paramMu) sum( min(max(vY - paramMu, 0), maxVal) ) - ballRadius;
% Should be the tightest boundaries
lowerBound = min(vY) - (maxVal / numElements); %<! Should give postive objective value
upperBound = max(vY); %<! Should give negative objective value

% Finding the root of 1D function within a segment
paramMu = MinimizeFunctionBiSection(hObjFun, lowerBound, upperBound, stopThr);

% Optimal solution
vX = min(max(vY - paramMu, 0), maxVal);


end

