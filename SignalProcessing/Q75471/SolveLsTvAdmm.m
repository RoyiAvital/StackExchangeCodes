function [ vX, mX ] = SolveLsTvAdmm( vX, mA, vY, mD, paramLambda, sSolverParams )
% ----------------------------------------------------------------------------------------------- %
%[ vX, mX ] = SolveLsTvAdmm( vX, mA, vB, mD, paramLambda, numIterations )
% Solve TV Regularized Least Squares Using Alternating Direction Method of
% Multipliers (ADMM) Method. Basically solves the problem given by:
% $$ \arg \min_{ x \in \mathbb{R}^{n} } \frac{1}{2} {\left\| A x - b \right|}_{2}^{2} + \lambda {\left\| D x \right\|}_{1} $$
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
%   - vY                -   Input Vector.
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
%   2.  Verified on MATLAB R2021a.
% Known Issues:
%   1.  C
% TODO:
%   1.  Add option to implement the `mD` operator as a function handler (To
%       be implemented by convolution).
% Release Notes:
%   -   1.0.000     04/06/2021  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    vX (:, 1) {mustBeFloat, mustBeReal}
    mA (:, :) {mustBeFloat, mustBeReal}
    vY (:, 1) {mustBeFloat, mustBeReal}
    mD (:, :) {mustBeFloat, mustBeReal}
    paramLambda (1, 1) {mustBeFloat, mustBeReal, mustBePositive}
    sSolverParams.paramRho (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 5
    sSolverParams.numIterations (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger} = 100
end

paramRho        = sSolverParams.paramRho;
numIterations   = sSolverParams.numIterations;

vAy = mA.' * vY;

% Decomposition of the constant matrix for faster solution of the linear
% system
mC = decomposition((mA.' * mA) + paramRho * (mD.' * mD), 'chol');

vZ = ProxL1(mD * vX, paramLambda / paramRho);
vU = mD * vX - vZ;

mX = zeros([size(vX, 1), numIterations]);
mX(:, 1) = vX;

for ii = 2:numIterations
    
    vX = mC \ (vAy + (paramRho * mD.' * (vZ - vU)));
    vZ = ProxL1(mD * vX + vU, paramLambda / paramRho);
    vU = vU + mD * vX - vZ;
    
    mX(:, ii) = vX;
    
end


end


function [ vX ] = ProxL1( vX, lambdaFactor )

% Soft Thresholding
vX = max(vX - lambdaFactor, 0) + min(vX + lambdaFactor, 0);


end

