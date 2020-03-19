function [ vX ] = ProxL2BallIndicatorL1NormReg( vY, paramMu )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProxL2BallIndicatorL1NormReg( vY, paramMu )
%   Solves g \left( x \right) = \mu {\left\| x \right\|}_{1} + {I}_{\left\| x \right\|}_2 \leq 1} \left( x \right)
%   using closed form solution.
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
%   1.  A
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     20/03/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

hSoftThreshold = @(vX, paramLambda) sign(vX) .* max(abs(vX) - paramLambda, 0);

vX = hSoftThreshold(vY, paramMu);
vX = vX / max(1, norm(vX));


end

