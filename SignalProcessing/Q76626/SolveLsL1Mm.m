function [ vX, mX ] = SolveLsL1Mm( vX, mA, vY, paramLambda, sSolverParams )
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
%   -   1.0.000     05/08/2021  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    vX (:, 1) {mustBeFloat, mustBeReal}
    mA (:, :) {mustBeFloat, mustBeReal}
    vY (:, 1) {mustBeFloat, mustBeReal}
    paramLambda (1, 1) {mustBeFloat, mustBeReal, mustBePositive}
    sSolverParams.numIterations (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger} = 100
end

epsVal = 1e-10;

numElements = size(vX, 1);

numIterations   = sSolverParams.numIterations;

mX = zeros(numElements, numIterations);

mAA = mA.' * mA;
vAy = mA.' * vY;

mX(:, 1) = vX;

for ii = 2:numIterations
    
    vX = (mAA + paramLambda * spdiags(1 ./ abs(vX + epsVal), 0, numElements, numElements)) \ vAy;
    
    mX(:, ii) = vX;
    
end


end

