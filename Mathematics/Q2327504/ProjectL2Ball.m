function [ vX ] = ProjectL2Ball( vY, ballRadius )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectL2Ball( vY, ballRadius, stopThr )
%   Solving the Orthogonal Projection Problem of the input vector onto the
%   L2 Ball.
% Input:
%   - vY            -   Input Vector.
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
%                       The projection of the Input Vector onto the L2
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

vX = min((ballRadius / norm(vY, 2)), 1) * vY;


end

