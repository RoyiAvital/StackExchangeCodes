function [ vX ] = ProxBoxIndicatorLInfNormReg( vY, paramAlpha )
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

hProxLInf = @(vX, paramLambda) vX - (paramLambda * ProjectL1Ball(vX / paramLambda, 1, 1e-6));

vX1 = min(max(vY, 0), 1);
vX1 = hProxLInf(vX1, paramAlpha);

valX1 = 0.5 * sum((vX1 - vY) .^ 2) + (paramAlpha * max(abs(vX1)));

vX2 = hProxLInf(vY, paramAlpha);
vX2 = min(max(vX2, 0), 1);

valX2 = 0.5 * sum((vX2 - vY) .^ 2) + (paramAlpha * max(abs(vX2)));

if(valX1 < valX2)
    vX = vX1;
else
    vX = vX2;
end


end

