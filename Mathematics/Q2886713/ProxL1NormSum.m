function [ vX ] = ProxL1NormSum( vY, paramGamma, paramB )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProxL1NormSum( vY, paramGamma, paramB )
%   Solving the Least Squares Problem with L1 Regularization (LASOO) with
%   Linear Equality Constraints (Sum of Elements) - Prox Operator.
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - paramGamma    -   Parameter Gamma - L2 Regularization Factor.
%                       Sets the coefficient of the L1 Term.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: [0, inf).
%   - paramB        -   Parameter B - Equality Constarint Parameter.
%                       Sets the scalar constarint of the equality
%                       condition.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - vX            -   Output Vector.
%                       The vector which minimizes the objective function
%                       and its sum of elements equals to paramB.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  Efficient Solvers for Sparse Subspace Clustering - https://arxiv.org/abs/1804.06291.
%   2.  https://math.stackexchange.com/a/2886715/33.
% Remarks:
%   1.  S
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     18/08/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

debugMode = OFF;

numElements = size(vY, 1);

vB      = sort([vY - paramGamma; vY + paramGamma], 'ascend');
iMin    = 1; %<! Bisection Lower Bound
iMax    = (2 * numElements) + 1; %<! Bisection Upper Bound
maxItr  = ceil(log2(iMax)) + 1;

if(debugMode == ON)
    vParamBeta = linspace(vB(1), vB(end), 1000);
    
    for ii = 1:length(vParamBeta)
        vZ(ii) = sum( ProxL1Norm(vY - vParamBeta(ii), paramGamma) ) - paramB;
    end
    for ii = 1:length(vB)
        vT(ii) = sum( ProxL1Norm(vY - vB(ii), paramGamma) ) - paramB;
    end
    
    figure();
    plot(vParamBeta, vZ);
    hold('on');
    plot(vB, vT, '*');
end

% Bisection
for ii = 1:maxItr
    if((iMax - iMin) <= 1)
        break;
    end
    idxJ = round((iMin + iMax) / 2);
    idxJ = min(max(idxJ, iMin + 1), iMax - 1);
    
    vX = ProxL1Norm(vY - vB(idxJ), paramGamma);
    if(sum(vX) > paramB)
        iMin = idxJ;
    else
        iMax = idxJ;
    end
end

if(debugMode == ON)
    plot(vB(iMin), sum( ProxL1Norm(vY - vB(iMin), paramGamma) ) - paramB, 'o');
    plot(vB(iMax), sum( ProxL1Norm(vY - vB(iMax), paramGamma) ) - paramB, 'o');
end

% Once the section is found there a closed form solution.
% Within the section [vB(iMin), vB(iMax)] the function has constant slope
% and for any paramBeta \in (vB(iMin), vB(iMax)) the function sign(vY -
% paramBeta) is constant.
% Hence sign(vY - paramBeta) .* max(abs(vY - paramBeta) - paramGamma), 0)
% equals to (Only at the support, where max(abs(vY - paramBeta) -
% paramGamma), 0) doesn't vanish) vY - paramBeta - paramGamam * sign().
% Looking when this sum equals to paramB happens:
% 1 / length(vSupport) * sumvY(vSupport) - 

paramBeta   = (vB(iMin) + vB(iMax)) / 2;
vX          = ProxL1Norm(vY - paramBeta, paramGamma);
vS          = (vX ~= 0);
paramBeta   = (sum(vY(vS) - paramGamma * sign(vX(vS))) - paramB) / sum(vS);   
vX          = ProxL1Norm(vY - paramBeta, paramGamma);


end


function [ vX ] = ProxL1Norm( vY, paramLambda )

vX = sign(vY) .* max(abs(vY) - paramLambda, 0);


end

