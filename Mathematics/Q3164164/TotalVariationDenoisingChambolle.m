function [ vX, vObjFunVal ] = TotalVariationDenoisingChambolle( vY, mD, paramLambda, stepSize, numIterations, hObjFun )
% ----------------------------------------------------------------------------------------------- %
% [ vX, vCostFunVal ] = TotalVariationDenoisingChambolle( vB, mD, lambdaFctr, stepSize, numIterations, hCostFun )
%   Solving Least Squares problem with the L1 norm of a linear operator
%   using Antonin Chambolle's method. The model (Objective Function) is given by:
%   \arg \min_x \frac{1}{2} {\left\| x - y \right\|}_{2}^{2} + \lambda {\left\| D x \right\|}_{1}
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mD            -   Linear Operator Matrix.
%                       Linear operator applied inside the L1 term. Usually
%                       it is a Sparse Matrix.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - paramLambda   -   Parameter Lambda.
%                       The parmater Lambda which factorize the L1 term.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
%   - stepSize      -   Step Size.
%                       The step size used in the iterative solver.
%                       Structure: Scalar. It is related to the Maximal
%                       Eigen Value of the Matrix mD.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
%   - numIterations -   Number of Iterations.
%                       Sets the number of iterations of the iterative
%                       solver. The number of iteration includes the
%                       creation of the initial state.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, 3, ...}.
%   - stepSize      -   Step Size.
%                       The step size used in the iterative solver.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
%   - hObjFun       -   Objective Function Handler.
%                       A function handler which gets as input the vector
%                       vX and calculates the objective function value of
%                       it.
%                       Structure: Function Hansdler (@(vX)).
%                       Type: Function Handler.
%                       Range: NA.
% Output:
%   - vX            -   Output Vector.
%                       The vector which minimizes the objective function.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vObjFunVal    -   Objective Function Value.
%                       The value of the Objective function for each
%                       iteration.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  An Algorithm for Total Variation Minimization and Applications - https://doi.org/10.1023/B:JMIV.0000011325.36760.1e.
% Remarks:
%   1.  The code implements the algorithm using Matrix Multiplication. Yet
%       in practice, many times it represents a 1D / 2D filter. Hence the
%       operation better be implemented using Convolution.
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     30/03/2019  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

mDD = mD * mD.';
vDY = mD * vY;

vP = zeros([size(mD, 1), 1]);

vObjFunVal = zeros([numIterations, 1]);

for ii = 1:numIterations
    vP = vP - (stepSize * ((paramLambda * paramLambda * mDD * vP) - (paramLambda * vDY)));
    vP = vP ./ (max(1, abs(vP)));
    
    vX = vY - paramLambda * mD.' * vP;
    
    vObjFunVal(ii) = hObjFun(vX);
end


end

