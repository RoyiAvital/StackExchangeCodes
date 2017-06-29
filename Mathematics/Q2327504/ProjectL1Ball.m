function [ vX ] = ProjectL1Ball( vY, ballRadius, stopThr )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectL1Ball( vY, ballRadius, stopThr )
%   Solving the Orthoginal Porjection Problem of the input vector onto the
%   L1 Ball using Dual Function and Newton Iteration.
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - ballRadius    -   Ball Radius.
%                       Sets the Radius of the L1 Ball. For Unit L1 Ball
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
%                       The projection of the Input Vector onto the L1
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
%   -   1.0.001     29/06/2017  Royi Avital
%       *   Enforcing Lambda to be non negative (Dealing with the case 'vY'
%           is obeying || vY ||_1 <= ballRadius).
%   -   1.0.000     27/06/2017  Royi Avital
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

% Enforcing paramLambda >= 0. Otherwise it suggests || vY ||_1 <= ballRadius.
% Hence the Optimal vX is given by vX = vY.
paramLambda = max(paramLambda, 0);

vX = sign(vY) .* max(abs(vY) - paramLambda, 0);


end

