function [ vX ] = ProjectSimplex( vY, ballRadius, stopThr )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectSimplex( vY, ballRadius, stopThr )
%   Solving the Orthoginal Porjection Problem of the input vector onto the
%   Simplex Ball using Dual Function and Newtin Iteration.
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - ballRadius    -   Ball Radius.
%                       Sets the Radiuf of the Simplex Ball. For Unit
%                       Simplex set to 1.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
%   - stopThr       -   Stopping Threshold.
%                       Sets the trheolds of the Newton Iteration. The
%                       absolute value of the Objective Function will be
%                       below the threshold.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
% Output:
%   - vX            -   Output Vector.
%                       The projection of the Input Vector onto the Simplex
%                       Ball.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  https://math.stackexchange.com/questions/2327504.
%   2.  https://en.wikipedia.org/wiki/Newton%27s_method.
% Remarks:
%   1.  a
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     09/05/2017  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

paramLambda = min(vY) - ballRadius;
% The objective functions which its root (The 'paramLambda' which makes it
% vanish) is the solution
objFun      = sum( max(vY - paramLambda, 0) ) - ballRadius;

while(abs(objFun) > stopThr)
    objFun      = sum( max(vY - paramLambda, 0) ) - ballRadius;
    df          = sum(-((vY - paramLambda) > 0)); %<! Derivative of 'objVal' with respect to Lambda
    paramLambda = paramLambda - (objFun / df); %<! Newton Iteration
end

vX = max(vY - paramLambda, 0);


end

