function [ vX, mX ] = SolveProxTvAdmm( vX, vY, mD, paramLambda, sSolverParams )
% ----------------------------------------------------------------------------------------------- %
%[ vX ] = SolveProxTvAdmm( vY, mD, paramLambda, numIterations )
% Solves the Prox of the Total Variation (TV) Norm using Alternating
% Direction Method of Multipliers (ADMM) Method.
% Basically solves the problem given by:
% $$ \arg \min_{ x \in \mathbb{R}^{n} } \frac{1}{2} {\left\| x - y \right|}_{2}^{2} + \lambda {\left\| D x \right\|}_{1} $$
% Input:
%   - vX                -   Optimization Vector.
%                           The vector to be optimized. Initialization of
%                           the iterative process.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vY                -   Measurements Vector.
%                           The model known data.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - mD                -   Model Matrix.
%                           The model matrix.
%                           Structure: Vector (m X n).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - paramLambda       -   Parameter Lambda.
%                           The L1 Regularization parameter.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - paramRho          -   The Rho Parameter.
%                           Sets the weight of the equality constraint.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range (0, inf).
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
%   2.  For high values of `paramLambda` it will require many iterations.
%   3.  The implementation support `mD` to be sparse matrix.
% Known Issues:
%   1.  C
% TODO:
%   1.  D
% Release Notes:
%   -   1.1.000     28/05/2021  Royi Avital
%       *   Updated to modern MATLAB (R2021a).
%   -   1.0.000     27/11/2019  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    vX (:, 1) {mustBeFloat, mustBeReal}
    vY (:, 1) {mustBeFloat, mustBeReal}
    mD (:, :) {mustBeFloat, mustBeReal}
    paramLambda (1, 1) {mustBeFloat, mustBeReal, mustBePositive}
    sSolverParams.paramRho (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 5
    sSolverParams.numIterations (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger} = 100
end

paramRho        = sSolverParams.paramRho;
numIterations   = sSolverParams.numIterations;

mX = zeros(size(vY, 1), numIterations);

mI = speye(size(vY, 1));
mC = decomposition(mI + paramRho * (mD.' * mD), 'chol');

vZ = ProxL1(mD * vX, paramLambda / paramRho);
vU = mD * vX - vZ;

mX(:, 1) = vX;

for ii = 2:numIterations
    
    vX = mC \ (vY + (paramRho * mD.' * (vZ - vU)));
    vZ = ProxL1(mD * vX + vU, paramLambda / paramRho);
    vU = vU + mD * vX - vZ;
    
    mX(:, ii) = vX;
    
end


end


function [ vX ] = ProxL1( vX, lambdaFactor )

% Soft Thresholding
vX = max(vX - lambdaFactor, 0) + min(vX + lambdaFactor, 0);


end

