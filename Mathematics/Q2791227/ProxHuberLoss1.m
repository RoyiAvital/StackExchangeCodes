function [ vX ] = ProxHuberLoss1( vY, paramLambda )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProxHuberLoss( vY, paramLambda )
%   Solves the Proximal Operator of the Huber Loss Function:
%   $$ \arg \min_{x} \frac{1}{2} {\left| x - y \right\|}_{2}^{2} + \lambda {H}_{1} \left( x \right) $$ 
%   Where $ {H}_{1} \left( x \right) $ is the Huber Loss Function with $
%   \delta = 1 $.
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
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
%   3.  Proximal Operator of the Huber Loss Function - https://math.stackexchange.com/questions/1650411.
% Remarks:
%   1.  This is the Proximal Operator of the Huber Loss Function for the
%       case \delta = 1.
%   2.  In the reference by Boyd they use $ \rho = 1 / \lambda $. Hence I
%       kept the refernce as is to show the code implements for the $
%       \lambda $ form of the Proximal Opertaor.
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

% For the form as in Boyd's book.
% paramLambda = 1 / paramLambda;
% The form as in Boyd's book.
% vX = ((paramLambda / (1 + paramLambda)) * vY) + ((1 / (1 + paramLambda)) * hProxL1(vY, 1 + (1 / paramLambda)));

vX = ((1 / (1 + paramLambda)) * vY) + ((paramLambda / (1 + paramLambda)) * hProxL1(vY, 1 + paramLambda));


end

