function [ vX ] = ProxHuberLoss( vY, paramDelta, paramLambda )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProxHuberLoss( vY, paramDelta, paramLambda )
%   Solves the Proximal Operator of the Huber Loss Function:
%   $$ \arg \min_{x} \frac{1}{2} {\left| x - y \right\|}_{2}^{2} + \lambda {H}_{\delta} \left( x \right) $$
%   Where $ {H}_{\delta} \left( x \right) $ is the Huber Loss Function.
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - paramDelta    -   Parameter Delta.
%                       The Delta Parameter of the Huber Loss Function.
%                       This is the value the Huber Loss changes from
%                       L2 Norm to the L1 Norm.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
%   - paramLambda   -   Parameter Lambda.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
% Output:
%   - vX            -   Output Vector.
%                       The solution of the Proximal Operator.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  Huber Loss (Wikipedia) - https://en.wikipedia.org/wiki/Huber_loss.
%   1.  Proximal Operator / Proximal Mapping of the Huber Loss Function - https://math.stackexchange.com/questions/3589025.
% Remarks:
%   1.  a
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     21/03/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

vX = vY - ((paramLambda * vY) ./ (max(abs(vY / paramDelta), paramLambda + 1)));


end

