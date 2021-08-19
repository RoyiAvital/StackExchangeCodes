function [ vX, mX ] = SolveLsL1Admm( vX, mA, vY, paramLambda, sSolverParams )
% ----------------------------------------------------------------------------------------------- %
%[ vX ] = SolveProxTvAdmm( vY, mD, paramLambda, numIterations )
% Solves the L1 Norm Regualrized Least Squares using Alternating
% Direction Method of Multipliers (ADMM) Method.
% Basically solves the problem given by:
% $$ \arg \min_{ x \in \mathbb{R}^{n} } \frac{1}{2} {\left\| A x - y \right|}_{2}^{2} + \lambda {\left\| x \right\|}_{1} $$
% Input:
%   - vX                -   Optimization Vector.
%                           The vector to be optimized. Initialization of
%                           the iterative process.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - mA                -   Input Matirx.
%                           The model matrix.
%                           Structure: Matrix (m X n).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vY                -   Measurements Vector.
%                           The model known data.
%                           Structure: Vector (m X 1).
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
    sSolverParams.paramRho (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 5
    sSolverParams.numIterations (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger} = 100
end

MAT_TYPE_SKINNY = 1;
MAT_TYPE_FAT    = 2;

if(size(mA, 1) >= size(mA, 2))
    matType = MAT_TYPE_SKINNY;
else
    matType = MAT_TYPE_FAT;
end

paramRho        = sSolverParams.paramRho;
numIterations   = sSolverParams.numIterations;

mX = zeros(size(mA, 2), numIterations);

% Caching facotirzation
mC = MatrixFactorize(mA, paramRho, matType);
vAy = mA.' * vY;

vZ = ProxL1(vX, paramLambda / paramRho);
vU = vX - vZ;

mX(:, 1) = vX;

for ii = 2:numIterations
    
    vQ = vAy + (paramRho * (vZ - vU));
    
    % Matrix Inversion Lemma
    switch(matType)
        case(MAT_TYPE_SKINNY)
            vX = mC \ vQ;
        case(MAT_TYPE_FAT)
            vX = (vQ / paramRho) - ((mA.' * (mC \ (mA * vQ))) / (paramRho * paramRho));
    end
    
    vZ = ProxL1(vX + vU, paramLambda / paramRho);
    vU = vU + vX - vZ;
    
    mX(:, ii) = vX;
    
end


end


function [ mC ] = MatrixFactorize( mA, paramRho, matType )

MAT_TYPE_SKINNY = 1;
MAT_TYPE_FAT    = 2;

switch(matType)
    case(MAT_TYPE_SKINNY)
        mI = speye(size(mA, 2));        
        mC = decomposition((mA.' * mA) + (paramRho * mI), 'chol');
    case(MAT_TYPE_FAT)
        mI = speye(size(mA, 1));
        mC = decomposition(mI + ((1 / paramRho) * (mA * mA.')), 'chol');
end


end


function [ vX ] = ProxL1( vX, lambdaFactor )

% Soft Thresholding
vX = max(vX - lambdaFactor, 0) + min(vX + lambdaFactor, 0);


end

