function [ vX ] = ProxHuberLoss( vY, paramLambda )
% ----------------------------------------------------------------------------------------------- %
% [ valLoss ] = HuberLoss( vX, paramDelta )
%   Applies the Huber Loss to the input vector.
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - ballRadius    -   Ball Radius.
%                       Sets the Radius of the L1 Ball. For Unit L1 Ball
%                       set to 1.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
% Output:
%   - vX            -   Output Vector.
%                       The projection of the Input Vector onto the L1
%                       Ball.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  Huber Loss (Wikipedia) - https://en.wikipedia.org/wiki/Huber_loss.
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

% For the form as in Boyd's book.
% paramLambda = 1 / paramLambda;

hProxL1 = @(vX, paramLambda) sign(vX) .* max(abs(vX) - paramLambda, 0);

% The form as in Boyd's book.
% vX = ((paramLambda / (1 + paramLambda)) * vY) + ((1 / (1 + paramLambda)) * hProxL1(vY, 1 + (1 / paramLambda)));

vX = ((1 / (1 + paramLambda)) * vY) + ((paramLambda / (1 + paramLambda)) * hProxL1(vY, 1 + paramLambda));


end

