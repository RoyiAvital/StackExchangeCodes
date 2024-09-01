function [ vX, paramLambda ] = SolveBp( mA, vB, paramEpsilon )
% ----------------------------------------------------------------------------------------------- %
%[ vX ] = SolveBpAdmm( mA, vB, paramLambda )
% Solve Basis Pursuit (Q1Eps) problem using ADMM.
% Input:
%   - mA                -   Input Matrix.
%                           The model matrix.
%                           Structure: Matrix (m X n).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vB                -   Input Vector.
%                           The model known data.
%                           Structure: Vector (m X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - paramEpsilon      -   Parameter Epsilon.
%                           Sets the threshold for the Least squares error
%                           of the solution.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: [0, inf).
% Output:
%   - vX                -   Output Vector.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - paramLambda       -   Parameter Lambda.
%                           Sets the balance between L1 minimization and
%                           Least Squares minimization.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: [0, inf).
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
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

SOLVER_METHOD_ADMM  = 1; %<! ADMM
SOLVER_METHOD_PPGM  = 2; %<! Proximal Gradient Method
SOLVER_METHOD_CD    = 3; %<! Coordinate Descent

solverMethod = SOLVER_METHOD_CD;

switch(solverMethod)
    case(SOLVER_METHOD_ADMM)
        hOptFun = @(paramLambda) (0.5 * sum(((mA * SolveBpAdmm(mA, vB, paramLambda)) - vB) .^ 2)) - paramEpsilon;
    case(SOLVER_METHOD_PPGM)
        hOptFun = @(paramLambda) (0.5 * sum(((mA * SolveLsL1ProxAccel(mA, vB, paramLambda, 200)) - vB) .^ 2)) - paramEpsilon;
    case(SOLVER_METHOD_CD)
        hOptFun = @(paramLambda) (0.5 * sum(((mA * SolveLsL1Cd(mA, vB, paramLambda, 200)) - vB) .^ 2)) - paramEpsilon;
end

sSolverOptions = optimset('Display', 'off');

paramLambda = fzero(hOptFun, [0.00001, 1000], sSolverOptions);

vX = SolveBpAdmm(mA, vB, paramLambda);


end

