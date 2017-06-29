function [ vX ] = ProjectLInfBall( vY, ballRadius )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectL2Ball( vY, ballRadius, stopThr )
%   Solving the Orthoginal Porjection Problem of the input vector onto the
%   L Inf Ball.
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - ballRadius    -   Ball Radius.
%                       Sets the Radius of the L Inf Ball. For Unit L Inf 
%                       Ball set to 1.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
% Output:
%   - vX            -   Output Vector.
%                       The projection of the Input Vector onto the L Inf
%                       Ball.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  h
% Remarks:
%   1.  a
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     29/06/2017  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

vX = sign(vY) .* min(abs(vY), ballRadius);


end

