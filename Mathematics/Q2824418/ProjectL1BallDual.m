function [ vX ] = ProjectL1BallDual( vY, ballRadius, vLowerBound, vUpperBound )
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
%   1.  https://math.stackexchange.com/a/2830242/33.
% Remarks:
%   1.  S
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

DEBUG_MODE = OFF;

paramLambda     = 0; %<! Initialization value

% Check feasibility of the problem
minSum = 0;
for ii = 1:size(vY, 1)
    if(sign(vLowerBound(ii)) == sign(vUpperBound(ii)))
        % Zero isn't within the boundary (Or both are zero)
        minSum = minSum + min(abs(vLowerBound(ii)), abs(vUpperBound(ii)));
    end
end

if(minSum > ballRadius)
    % The problem is infeasible
    vX = mean([vLowerBound, vUpperBound], 2);
    return;
end

% The dual objective function which should be maximized to find the optimal
% 'paramLambda'.
% The functios is negated as we want to maximize while using
% MATLAB's minimization function
hObjFun = @(paramLambda) -ObjectiveDualFunction(vY, ballRadius, vLowerBound, vUpperBound, paramLambda); %<! Objective function

if(DEBUG_MODE == ON)
    numSamples = 1000;
    
    vParamLambda    = linspace(0, 5, numSamples);
    vObjVal         = zeros([numSamples, 1]);
    
    for ii = 1:numSamples
        vObjVal(ii) = hObjFun(vParamLambda(ii));
    end
    
    figure();
    plot(vParamLambda, vObjVal);
end

paramLambda = fminsearch(hObjFun, paramLambda); %<! Objective Function isn't smooth hence can't use Newton Method
paramLambda = max(paramLambda, 0);

% Solution as derived for the updated forumaltion of the problem.
[~, vX] = ObjectiveDualFunction(vY, ballRadius, vLowerBound, vUpperBound, paramLambda);


end


function [ valObj, vX ] = ObjectiveDualFunction( vY, ballRadius, vL, vU, paramLambda )

vX      = zeros([size(vY, 1), 1]);

for ii = 1:size(vY, 1)
    
    if(sign(vL(ii)) == sign(vU(ii)))
        valX = ProjBoxFunction(vY(ii) - (paramLambda * sign(vL(ii))), vL(ii), vU(ii));
    elseif(sign(vY(ii)) == sign(vL(ii)))
        % Implictily vL(ii) <= 0, vU(ii) >= 0 and vY(ii) <= 0
        valX = ProjBoxFunction(vY(ii) + paramLambda, vL(ii), 0); %<! Making sure sign of valX is non positive
    elseif(sign(vY(ii)) == sign(vU(ii)))
        % Implictily vU(ii) >= 0, vL(ii) <= 0 and vY(ii) >= 0
        valX = ProjBoxFunction(vY(ii) - paramLambda, 0, vU(ii)); %<! Making sure sign of valX is non negative
    end
    
    vX(ii) = valX;
    
end

valObj = (0.5 * sum((vX - vY) .^ 2)) + (paramLambda * (sum(abs(vX)) - ballRadius));


end


function [ outVal ] = ProjBoxFunction( valIn, valL, valU )

outVal = min(max(valIn, valL), valU);


end


