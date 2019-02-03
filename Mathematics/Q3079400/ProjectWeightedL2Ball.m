function [ vX ] = ProjectWeightedL2Ball( vY, mW, vC, ballRadius )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectWeightedL2Ball( vY, mW, vC, ballRadius, stopThr )
%   Solving the Orthogonal Porjection Problem of the input vector onto the
%   Weighted L2 Ball. The problem is given by: 
%   \arg \min_x \frac{1}{2} {\left\| x - y \right\|}_{2}^{2}
%   subject to {(x - c)}^{T} W (x - c) \leq r
%   Where W is a PSD matrix (Hence the "Ball" is an ellipse).
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mW            -   Weighing Matrix.
%                       The PSD Matrix which is the weights of the L2 Norm.
%                       Basically creates an ellipse in space.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vC            -   Center Vector.
%                       The center of the ball (Ellipse).
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - ballRadius    -   Ball Radius.
%                       Sets the Radius of the L2 Ball. For Unit L2 Ball
%                       set to 1.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
% Output:
%   - vX            -   Output Vector.
%                       The projection of the Input Vector onto the
%                       Weighted L2 Ball.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  A
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     03/02/2019  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

if( (vY - vC).' * mW * (vY - vC) <= ballRadius )
    vX = vY;
    return;
end

mI = eye(size(vY, 1));
% The optimal vX as a function of paramLambda.
hCalcX = @(paramLambda) (mI + (2 * paramLambda * mW)) \ (vY + (2 * paramLambda * mW * vC));

% The objective function where its zero is the optimal paramLambda
hObjFun     = @(paramLambda) ((hCalcX(paramLambda) - vC).' * mW * (hCalcX(paramLambda) - vC)) - ballRadius;
paramLambda = fzero(hObjFun, 0);
vX          = hCalcX(paramLambda);


end

