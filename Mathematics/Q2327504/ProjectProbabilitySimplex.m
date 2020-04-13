function [ vX ] = ProjectProbabilitySimplex( vY, ballRadius )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectProbabilitySimplex( vY )
%   Solving the Orthogonal Projection Problem of the input vector onto the
%   Probability (Unit) Simplex. This solver returns the analytic exact
%   solution.
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - vX            -   Output Vector.
%                       The projection of the Input Vector onto the Simplex
%                       Ball.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  Projection Onto A Simplex (https://arxiv.org/abs/1101.6081).
% Remarks:
%   1.  This implementation extends the paper 'Projection Onto A Simplex'
%       with support for any arbitrary postive ball radius.
% TODO:
%   1.  C
% Release Notes:
%   -   1.2.000     13/04/2020  Royi Avital
%       *   Added support for different Ball Radius. The Probability
%           Simplex is given by 'ballRadius = 1'.
%   -   1.0.000     13/04/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numElements = size(vY, 1);
vYY         = sort(vY, 'ascend'); %<! Sorted values

valT = (sum(vY) - ballRadius) / numElements;

for ii = (numElements - 1):-1:1
    valTT = (sum(vYY((ii + 1):numElements)) - ballRadius) / (numElements - ii);
    if( valTT > vYY(ii) )
        valT = valTT;
        break;
    end
end

vX = max(vY - valT, 0);


end

