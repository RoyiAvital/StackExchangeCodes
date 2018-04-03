function [ vX ] = SolveBpAdmm( mA, vB, paramLambda )
% ----------------------------------------------------------------------------------------------- %
%[ vX ] = SolveBpAdmm( mA, vB, paramLambda )
% Solve Basis Pursuit (Q1) problem using ADMM.
% Input:
%   - mA                -   Input Matirx.
%                           The model matrix.
%                           Structure: Matrix (m X n).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vB                -   Input Vector.
%                           The model known data.
%                           Structure: Vector (m X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - paramLambda       -   Parameter Lambda.
%                           Sets the balance between L1 minimization and
%                           Least Squares minimization.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
% Output:
%   - vX                -   Output Vector.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References
%   1.  A
% Remarks:
%   1.  A
% Known Issues:
%   1.  A
% TODO:
%   1.  B
% Release Notes:
%   -   1.0.000     03/04/2018
%       *   First realease version.
% ----------------------------------------------------------------------------------------------- %

numIterations = 350;
primalEps = 1e-6; %<! Stopping Condition

numRows = size(mA, 1);
numCols = size(mA, 2);

vX = zeros([numCols, 1]);
vU = zeros([numCols, 1]);
vV = zeros([numCols, 1]);

mAA = mA.' * mA;
vAb = mA.' * vB;
mI = eye(numCols);

mAAI = mAA + mI;

mL = chol(mAAI, 'lower');
mU = mL.';

sL.LT = true();
sU.UT = true();

for ii = 1:numIterations
    % vX = mAAI \ (vAb + vV - vU);
    vX = linsolve(mU, linsolve(mL, vAb + vV - vU, sL), sU); %<! https://www.mathworks.com/help/matlab/ref/linsolve.html
    vV = SoftThresholding(vX + vU, paramLambda);
    vU = vU + vX - vV;
    
%     if(sqrt(mean((vX - vV) .^ 2)) < primalEps) %<! Primal Convergence
%         break;
%     end
end


end


function [ vX ] = SoftThresholding( vX, lambdaFactor )

% Soft Thresholding
vX = max(vX - lambdaFactor, 0) + min(vX + lambdaFactor, 0);


end

