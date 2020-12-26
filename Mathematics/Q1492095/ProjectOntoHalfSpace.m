function [ vX ] = ProjectOntoHalfSpace( vY, vA, valB )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectOntoHalfSpace( vY, vA, valB )
%   Projects onto the half space defined by vA.' * vY <= vB. 
% Input:
%   - vY            -   Input Vector.
%                       Input vector to be projected.
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vA            -   Half Space Vector.
%                       The vector which defines the half space.
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - valB          -   Half Spave Scalar.
%                       The scalar which defines the half space.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - vX            -   Solution Vector.
%                       The projected vector.
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  Orthogonal Projection onto a Half Space - https://math.stackexchange.com/questions/318740.
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

valR = (vA.' * vY) - valB;

if(valR > 0)
    vX = vY - ((valR / (vA.' * vA)) * vA);
else
    vX = vY;
end


end

