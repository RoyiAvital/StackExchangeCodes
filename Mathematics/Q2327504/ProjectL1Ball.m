function [ vX ] = ProjectL1Ball( vY, ballRadius, stopThr )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectL1Ball( vY, ballRadius, stopThr )
%   Solving the Orthoginal Porjection Problem of the input vector onto the
%   L1 Ball using Dual Function and Newtin Iteration.
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - ballRadius    -   Ball Radius.
%                       Sets the Radiuf of the L1 Ball. For Unit L1 Ball
%                       set to 1.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
%   - stopThr       -   Stopping Threshold.
%                       Sets the trheold of the Newton Iteration. The
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

paramLambda     = 0;
% The objective functions which its root (The 'paramLambda' which makes it
% vanish) is the solution
objVal          = sum(max(abs(vY) - paramLambda, 0)) - ballRadius;

while(abs(objVal) > stopThr)
    objVal          = sum(max(abs(vY) - paramLambda, 0)) - ballRadius;
    df              = sum(-((abs(vY) - paramLambda) > 0)); %<! Derivative of 'objVal' with respect to Lambda
    paramLambda     = paramLambda - (objVal / df); %<! Newton Iteration
end

vX = sign(vY) .* max(abs(vY) - paramLambda, 0);


end

