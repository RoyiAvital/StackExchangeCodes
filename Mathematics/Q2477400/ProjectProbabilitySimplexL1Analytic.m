function [ vX ] = ProjectProbabilitySimplexL1Analytic( vY )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectProbabilitySimplexL1( vY )
%   Solves \arg \min_{x} || x - y ||_{1} s.t. x \in Probability Simplex.
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (n x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - vX            -   Solution Vector.
%                       The solution to the optimization problem.
%                       Structure: Vector (n x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  See https://math.stackexchange.com/questions/2477400.
% Remarks:
%   1.  The sconcept is this is an element wise optimizatio. so first for
%       all values of 'vY' the closest 'vX' can get is 0. Than for the
%       rest, they either sum to 1, then take them as they are, or they are
%       above or below 1. In any case there is a busget to spread in order
%       to have the sum of 'vX' to be 1.
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     13/04/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numElements = size(vY, 1);

vX = zeros(numElements, 1);

vS = (vY > 0);
vI = find(vS);
sumS = sum(vY(vI));

vX(vI) = vY(vI);

% cover the cases 'vY(vI)' doesn't add up to 1
if(sumS > 1)
    resX = sumS - 1; %<! Residual for Sum of 1
    for ii = vI.'
        minVal = min(vY(ii), resX); %<! Taking as much as we can from the budget but keeping the element non negative
        vX(ii) = vY(ii) - minVal;
        resX = resX - minVal;
        if(resX == 0)
            % The budget is empty, 'sum(vX) == 1'.
            break;
        end
    end
elseif(sumS < 1)
    % vX(vI) = vX(vI) + ((1 - sumS) / sum(vS)); %<! Spread all over elements in 'vS'
    % vX(:) = vX + ((1 - sumS) / numElements);%<! Spread all over elements of 'vX'
    % vX(vI(1)) = vX(vI(1)) + (1 - sumS); %<! All into one (Safe as any element is farther or as far [Single Value] from 1 than '1 - sumS')
    vX(1) = vX(1) + (1 - sumS); %<! All into one (Safe as any element is farther or as far [Single Value] from 1 than '1 - sumS')
end


end

