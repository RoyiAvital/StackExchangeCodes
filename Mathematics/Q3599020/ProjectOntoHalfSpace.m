function [ vX ] = ProjectOntoHalfSpace( vY, vA, valB )
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

