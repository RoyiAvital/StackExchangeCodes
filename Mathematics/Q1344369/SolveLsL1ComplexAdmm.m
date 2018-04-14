function [ vX, mX ] = SolveLsL1ComplexAdmm( mA, vB, paramLambda, numIterations )
% ----------------------------------------------------------------------------------------------- %
%[ vX, mX ] = SolveLsL1ComplexSubGrad( mA, vB, lambdaFctr, numIterations )
% Solves the 0.5 * || A x - b ||_2 + \lambda || x ||_1 problem using ADMM
% Method. The model allows A, b and x to be Complex.
% Input:
%   - mA                -   Model Matrix.
%                           The model matrix.
%                           Structure: Matrix (m X n).
%                           Type: 'Single' / 'Double' (Complex).
%                           Range: (-inf, inf).
%   - vB                -   Input Vector.
%                           The model known data.
%                           Structure: Vector (m X 1).
%                           Type: 'Single' / 'Double' (Complex).
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
%   1.  Wikipedia ADMM Method - https://en.wikipedia.org/wiki/Augmented_Lagrangian_method#Alternating_direction_method_of_multipliers.
% Remarks:
%   1.  This implementation assumes the Rho Factor is 1.
% Known Issues:
%   1.  A
% TODO:
%   1.  B
% Release Notes:
%   -   1.0.000     07/11/2016
%       *   First realease version.
% ----------------------------------------------------------------------------------------------- %

numRows = size(mA, 1);
numCols = size(mA, 2);

vX  = pinv(mA) * vB; %<! Dealing with "Fat Matrix"
vU = zeros([numCols, 1]);
vV = zeros([numCols, 1]);

mAA = mA' * mA;
vAb = mA' * vB;
mI = eye(numCols);

mAAI = mAA + mI;

mL = chol(mAAI, 'lower');
mU = mL';

sL.LT = true();
sU.UT = true();

mX = zeros([numCols, numIterations]);
mX(:, 1) = vX;

for ii = 2:numIterations
    % vX = mAAI \ (vAb + vV - vU);
    vX = linsolve(mU, linsolve(mL, vAb + vV - vU, sL), sU); %<! https://www.mathworks.com/help/matlab/ref/linsolve.html
    vV = ProxL1(vX + vU, paramLambda);
    vU = vU + vX - vV;
    
    mX(:, ii) = vX;
end


end


function [ vX ] = ProxL1( vX, lambdaFactor )

% Soft Thresholding - Complex Domain -> Keep Phase, Soft Threshold the
% Modulus

% vXAbs   = abs(vX);
% vXPhase = angle(vX);
% 
% vX = max(vXAbs - lambdaFactor, 0) .* exp(1i * vXPhase);

vXAbs = abs(vX);

vX              = (vX ./ vXAbs) .* max((vXAbs - lambdaFactor), 0);
vX(vXAbs == 0)  = 0;


end

