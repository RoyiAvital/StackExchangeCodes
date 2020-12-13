function [ vX, mX ] = SolveLsTvAdmm( vX, mA, vB, mD, paramLambda, numIterations )
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

MAT_TYPE_SKINNY = 1;
MAT_TYPE_FAT    = 2;

if(size(mA, 1) >= size(mA, 2))
    matType = MAT_TYPE_SKINNY;
else
    matType = MAT_TYPE_FAT;
end

vAb = mA.' * vB;
if(isempty(vX))
    vX  = pinv(mA) * vB; %<! Dealing with "Fat Matrix"
end

paramRho = 5;

% mC = inv((mA.' * mA) + paramRho * (mD.' * mD));
mC = decomposition((mA.' * mA) + paramRho * (mD.' * mD), 'chol');

vZ = ProxL1(mD * vX, paramLambda / paramRho);
vU = mD * vX - vZ;

mX = zeros([size(vX, 1), numIterations]);
mX(:, 1) = vX;

for ii = 2:numIterations
    
    vX = mC \ (vAb + (paramRho * mD.' * (vZ - vU)));
    vZ = ProxL1(mD * vX + vU, paramLambda / paramRho);
    vU = vU + mD * vX - vZ;
    
    mX(:, ii) = vX;
    
end


end


function [ vX ] = ProxL1( vX, lambdaFactor )

% Soft Thresholding
vX = max(vX - lambdaFactor, 0) + min(vX + lambdaFactor, 0);


end


function [ mL, mU ] = MatrixFactorize( mA, paramRho, matType )

MAT_TYPE_SKINNY = 1;
MAT_TYPE_FAT    = 2;

switch(matType)
    case(MAT_TYPE_SKINNY)
        mI = eye(size(mA, 2));
        mL = chol((mA.' * mA) + (paramRho * mI), 'lower');
    case(MAT_TYPE_FAT)
        mI = eye(size(mA, 1));
        mL = chol(mI + ((1 / paramRho) * (mA * mA.')), 'lower');
end

mU = mL.';


end

