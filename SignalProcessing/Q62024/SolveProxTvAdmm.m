function [ vX ] = SolveProxTvAdmm( vY, mD, paramLambda, numIterations )
% ----------------------------------------------------------------------------------------------- %
%[ vX ] = SolveProxTvAdmm( vY, mD, paramLambda, numIterations )
% Solves the Prox of the Total Variation (TV) Norm using Alternating
% Direction Method of Multipliers (ADMM) Method.
% Basically solves the problem given by:
% $$ \arg \min_{ x \in \mathbb{R}^{n} } \frac{1}{2} {\left\| x - y \right|}_{2}^{2} + \lambda {\left\| D x \right\|}_{1} $$
% Input:
%   - vX                -   input Vector.
%                           Initialization of the iterative process.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - mA                -   Input Matirx.
%                           The model matrix.
%                           Structure: Matrix (m X n).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vB                -   input Vector.
%                           The model known data.
%                           Structure: Vector (m X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - paramLambda       -   Parameter Lambda.
%                           The L1 Regularization parameter.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - numIterations     -   Number of Iterations.
%                           Number of iterations of the algorithm.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range {1, 2, ...}.
% Output:
%   - vX                -   Output Vector.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References
%   1.  Wikipedia ADMM - https://en.wikipedia.org/wiki/Augmented_Lagrangian_method#Alternating_direction_method_of_multipliers.
% Remarks:
%   1.  Using vanilla ADMM with no optimization of the parameter or
%       smoothing.
%   2.  Matrix Factorization caching according to "Matrix Inversion Lemma"
%       (See S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein -
%       Distributed Optimization and Statistical Learning via the
%       Alternating Direction Method of Multipliers Page 28). Basically:
%       (mA.' * mA + paramRho * I)^(-1) = (1 / paramRho) + (1 / (paramRho * paramRho)) * mA.' * (I + (1 /
%       paramRho) * mA * mA.')^(-1) * mA
% Known Issues:
%   1.  A
% TODO:
%   1.  Pre calculate decomposition of the Linear System.
% Release Notes:
%   -   1.0.000     27/11/2019  Royi Avital
%       *   First realease version.
% ----------------------------------------------------------------------------------------------- %

paramRho = 5;

mI = eye(size(vY, 1));
mC = decomposition(mI + paramRho * (mD.' * mD), 'chol');

vX = vY;
vZ = ProxL1(mD * vX, paramLambda / paramRho);
vU = mD * vX - vZ;

for ii = 2:numIterations
    
    vX = mC \ (vY + (paramRho * mD.' * (vZ - vU)));
    vZ = ProxL1(mD * vX + vU, paramLambda / paramRho);
    vU = vU + mD * vX - vZ;
    
end


end


function [ vX ] = ProxL1( vX, lambdaFactor )

% Soft Thresholding
vX = max(vX - lambdaFactor, 0) + min(vX + lambdaFactor, 0);


end

