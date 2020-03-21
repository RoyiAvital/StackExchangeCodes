function [ vX ] = ProxHuberLossBoyd( vY, paramDelta, paramLambda )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProxHuberLossBoyd( vY, paramDelta, paramLambda )
%   Solves the Proximal Operator of the Huber Loss Function:
%   $$ \arg \min_{x} \frac{1}{2} {\left| x - y \right\|}_{2}^{2} + \lambda {H}_{\delta} \left( x \right) $$
%   Where {H}_{\delta} \left( x \right) is the Huber Loss Function.
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
%   2.  Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers (See Huber Fitting).
%   3.  Proximal Operator of the Huber Loss Function - https://math.stackexchange.com/questions/3589025.
% Remarks:
%   1.  In the reference by Boyd they use the $ \rho = 1 / \lambda $ form of
%       the Proximal Operator. Hence the adoption of the scaling. Also the
%       book use Huber Loss Function with $ \delta = 1 $. 
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

hProxL1 = @(vX, paramLambda) sign(vX) .* max(abs(vX) - paramLambda, 0);
vX = ((1 / (1 + paramLambda)) * vY) + (paramDelta * (paramLambda / (1 + paramLambda)) * hProxL1((vY / paramDelta), 1 + paramLambda));


end

