function [ vX, mX ] = SolveLsL1Admm( mA, vB, paramLambda, numIterations )
% ----------------------------------------------------------------------------------------------- %
%[ vX, mX ] = SolveLsL1Admm( mA, vB, lambdaFctr, numIterations )
% Solve L1 Regularized Least Squares Using Alternating Direction Method of Multipliers (ADMM) Method.
% Input:
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
%   -   1.1.000     23/08/2017
%       *   Added optimized factorization.
%   -   1.0.000     23/08/2017
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
vX  = pinv(mA) * vB; %<! Dealing with "Fat Matrix"

paramRho    = 5;

% Cahing Factorization
[mL, mU] = MatrixFactorize(mA, paramRho, matType);

vZ = vX;
vU = vX;

mX = zeros([size(vX, 1), numIterations]);
vX = mX(:, 1);

for ii = 2:numIterations
    
    vQ = (vAb + (paramRho * vZ) - vU);
    
    % Matrix Inversion Lemma
    switch(matType)
        case(MAT_TYPE_SKINNY)
            vX = mU \ (mL \ vQ);
        case(MAT_TYPE_FAT)
            vX = (vQ / paramRho) - ((mA.' * (mU \ (mL \ (mA * vQ)))) / (paramRho * paramRho));
    end
    
    % This doesn't work.
    % vX = max(vX, 0); %<! Project onto R+
    
    vZ = ProxL1(vX + (vU / paramRho), paramLambda / paramRho);
    % Projectin z. See Dual methods and ADMM (http://www.stat.cmu.edu/~ryantibs/convexopt-F13/lectures/23-dual-meth.pdf) Pg. 10.
    vZ = max(vZ, 0); %<! Project onto R+
    
    vU = vU + (paramRho * (vX - vZ));
    
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

