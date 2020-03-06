function [ vX ] = ProxLogisticLossFunctionGd( vX, vY, vC, paramLambda, numIterations, stopThr )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProxLogisticLossFunction( vX, vY, vC, paramLambda, numIterations, stopThr )
%   Calculates the Prox of the Logistic Cost Function using Gradient Descent.
% Input:
%   - vX            -   Input Vector.
%                       Starting point for the iterative procedure.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vY            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vC            -   Model Vector.
%                       The model vector in the Logsitic Cost Function.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - numIterations -   Number of Iterations.
%                       Sets the number of iterations for the algorithm to
%                       run.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range {1, 2, ...}.
%   - stopTol       -   Stopping Condition Tolerance.
%                       Sets the stopping threshold for the L Inf (Maximum
%                       Absolute Value) of the change between 2 iterations
%                       of the algorithm.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range [0, inf).
% Output:
%   - vX            -   Output Vector.
%                       The Prox for the logistic cost function for the
%                       input vector 'vY'.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  https://math.stackexchange.com/a/3571521/33.
%   2.  Elementary Numerical Analysis MATH:3800/CS:3700(22M:072/22C:072)
%       (https://homepage.divms.uiowa.edu/~whan/3800.d/3800.html, Section 3.4).
% Remarks:
%   1.  On some cases it fails to converge (While Newton Method converge to
%       the right solution).
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     06/03/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

vXPrev = vX;
vG = zeros(size(vX, 1), 1);

hObjFun = @(vX) 0.5 * sum((vX - vY) .^ 2) + (paramLambda * log(1 + exp(-vC.' * vX)));
stepSizeMax = 2;
sSolverOptions = optimset('fminbnd');
sSolverOptions = optimset(sSolverOptions, 'Display', 'off');

for ii = 1: numIterations
    vXPrev(:) = vX;
    
    valExp  = exp(-vC.' * vX);
    vG(:) = vX - vY - (paramLambda * (valExp / (1 + valExp)) * vC);
    
    hStepSizeFun    = @(stepSize) hObjFun(vX - (stepSize * vG));
    stepSize        = fminbnd(hStepSizeFun, 0, stepSizeMax, sSolverOptions);
    
    vX(:) = vX - (stepSize * vG);
    
    if(max(abs(vX - vXPrev)) < stopThr)
        break;
    end
end


end

