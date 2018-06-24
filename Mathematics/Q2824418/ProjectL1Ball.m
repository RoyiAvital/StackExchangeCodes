function [ vX ] = ProjectL1Ball( vY, ballRadius, vLowerBound, vUpperBound )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectL1Ball( vY, ballRadius, vLowerBound, vUpperBound )
%   Solving the Orthogonal Porjection Problem of the input vector onto the
%   L1 Ball with Box Constraints using Dual Function.
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
%   - vLowerBound   -   Lower Bound Vector.
%                       Sets the lower bound values of the solution
%                       (Element wise).
%                       Structure: Vector.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vUpperBound   -   Upper Bound Vector.
%                       Sets the upper bound values of the solution
%                       (Element wise).
%                       Structure: Vector.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - vX            -   Output Vector.
%                       The projection of the Input Vector onto the L1
%                       Ball.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  https://math.stackexchange.com/a/2829935/33.
% Remarks:
%   1.  Solution is based on the trick || x ||_{1} = sign(x)^T * x. Hence
%       reformulating the problem for easy solution.
%   2.  Boundaries must be updated to validate the Trick for the solution.
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     24/06/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

paramLambda     = 0; %<! Initialization value

if(sum(min(abs(vLowerBound), abs(vUpperBound))) > ballRadius)
    % The problem is infeasible
    vX = mean([vLowerBound, vUpperBound], 2);
    return;
end


% The objective functions which its root (The 'paramLambda' which makes it
% vanish) is the solution

% The vector vS should be vS = sign(vX). Using some basic logic the sign of
% the solution could be infered without having it explicitly. By default it
% is given by the initialization
vS = sign(vY);

for ii = 1:size(vY, 1)
    if(sign(vLowerBound(ii)) == sign(vUpperBound(ii)))
        vS(ii) = sign(vLowerBound(ii));
    elseif(sign(vY(ii)) == sign(vLowerBound(ii)))
        % Updating boundaries to make sure the solution for vX has the
        % correct sign.
        vUpperBound(ii) = 0;
    elseif(sign(vY(ii)) == sign(vUpperBound(ii)))
        % Updating boundaries to make sure the solution for vX has the
        % correct sign.
        vLowerBound(ii) = 0;
    end
    
end

hProjFun    = @(vX) min(max(vX, vLowerBound), vUpperBound); %<! Validated projection function
hObjFun     = @(paramLambda) (vS.' * hProjFun(vY - (paramLambda * vS))) - ballRadius; %<! Objective function

if(DEBUG_MODE == ON)
    numSamples = 1000;
    
    vParamLambda    = linspace(0, 5, numSamples);
    vObjVal         = zeros([numSamples, 1]);
    
    for ii = 1:numSamples
        vObjVal(ii) = (vS.' * hProjFun(vY - (vParamLambda(ii) * vS))) - ballRadius;
    end
    
    figure();
    plot(vParamLambda, vObjVal);
end

paramLambda = fzero(hObjFun, paramLambda); %<! Objective Function isn't smooth hence can't use Newotin Method
paramLambda = max(paramLambda, 0);

% Solution as derived for the updated forumaltion of the problem.
vX = hProjFun(vY - (paramLambda * vS));


end

