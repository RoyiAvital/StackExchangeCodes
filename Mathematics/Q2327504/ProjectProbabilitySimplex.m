function [ vX ] = ProjectProbabilitySimplex( vY )
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
%   1.  B
% TODO:
%   1.  Explore options to add support for other Radius of the Simplex
%       (Instead of sum to 1 sum to some other positive number).
% Release Notes:
%   -   1.0.000     13/04/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numElements = size(vY, 1);
vYY         = sort(vY, 'ascend'); %<! Sorted values

valT = (sum(vY) - 1) / numElements;

for ii = (numElements - 1):-1:1
    valTT = (sum(vYY((ii + 1):numElements)) - 1) / (numElements - ii);
    if( valTT > vYY(ii) )
        valT = valTT;
        break;
    end
end

vX = max(vY - valT, 0);


end

