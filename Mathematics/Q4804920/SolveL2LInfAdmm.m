function [ vX, mX ] = SolveL2LInfAdmm( vX, mA, vY, paramLambda, paramRho, numIterations )
% ----------------------------------------------------------------------------------------------- %
%[ vX, mX ] = SolveLsL1Admm( mA, vB, lambdaFctr, numIterations )
% Solve LInf Regularized Least Squares Using Alternating Direction Method of Multipliers (ADMM) Method.
% Input:
%   - vX                -   Input Vector.
%                           Initialization of the solution.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - mA                -   Input Matirx.
%                           The model matrix.
%                           Structure: Matrix (m X n).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vY                -   input Vector.
%                           The model known data.
%                           Structure: Vector (m X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - paramLambda       -   Parameter Lambda.
%                           The LInf Regularization parameter.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - paramRho          -   Parameter Rho.
%                           The Augmented Lagrange parameter.
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
%   - mX                -   Solution Path Matrix.
%                           Structure: Vector (n X numIterations).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References
%   1.  Wikipedia ADMM - https://en.wikipedia.org/wiki/Augmented_Lagrangian_method#Alternating_direction_method_of_multipliers.
% Remarks:
%   1.  Using vanilla ADMM with no optimization of the parameter or
%       smoothing.
% Known Issues:
%   1.  A
% TODO:
%   1.  Auto stopping criteria.
%   1.  Initialization of `vZ` and `vU`.
% Release Notes:
%   -   1.0.000     14/11/2023
%       *   First realease version.
% ----------------------------------------------------------------------------------------------- %

arguments (Input)
    vX (:, 1) {mustBeNumeric, mustBeVector, mustBeFinite, mustBeFloat}
    mA (:, :) {mustBeNumeric, mustBeFinite, mustBeFloat}
    vY (:, 1) {mustBeNumeric, mustBeVector, mustBeFinite, mustBeFloat}
    paramLambda (1, 1) {mustBeNumeric, mustBeNonnegative, mustBeFinite, mustBeFloat}
    paramRho (1, 1) {mustBeNumeric, mustBeNonnegative, mustBeFinite, mustBeFloat} = 5
    numIterations (1, 1) {mustBeInteger, mustBeNonnegative} = 100
end

arguments (Output)
    vX (:, 1) {mustBeNumeric, mustBeVector, mustBeFinite, mustBeFloat}
    mX (:, :) {mustBeNumeric, mustBeFinite, mustBeFloat}
end

numElements = size(vX, 1);

% Cahing Factorization
mC = decomposition(eye(numElements) + paramRho * (mA' * mA), 'chol');

vZ = vX;
vU = vX;

mX = zeros([numElements, numIterations]);
vX = mX(:, 1);

for ii = 2:numIterations
    
    vX = paramRho * (mC \ (mA' * (vZ + vY - vU)));
    vZ = ProxLInf(mA * vX - vY + vU, paramLambda / paramRho);
    vU = vU + (mA * vX) - vZ - vY;
    
    mX(:, ii) = vX;
    
end


end


function [ vX ] = ProxLInf( vX, paramLambda )

vX = vX - paramLambda * ProjectL1BallExact(vX ./ paramLambda, 1);


end

