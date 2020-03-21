function [ valLoss ] = HuberLoss( vX, paramDelta )
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

numElements = length(vX);

valLoss = 0;

for ii = 1:numElements
    if(abs(vX(ii)) <= paramDelta)
        valLoss = valLoss + (0.5 * vX(ii) * vX(ii));
    else
        valLoss = valLoss + (paramDelta * (abs(vX(ii)) - (0.5 * paramDelta)));
    end
    
end


end

