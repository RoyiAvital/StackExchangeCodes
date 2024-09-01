function [ vX ] = SolveLsNormConst( mA, vB, normConst )
% ----------------------------------------------------------------------------------------------- %
%[ vX ] = SolveLsNormConst( mA, vB, normConst )
% Solves norm constrained Least Squares problem by finding the optimal Dual
% Variable (paramLambda) of the KKT Conditions by successively solving
% Tikhonov Regularized Least Squares problems.
% Input:
%   - mA                -   Input Matrix.
%                           The given matrix of the problem 0.5 || A x - b|| ^ 2.
%                           Structure: Vector (Column Vector).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vB                -   Input Vector.
%                           The given vector of the problem 0.5 || A x - b|| ^ 2.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, ...}.
%   - normConst         -   Norm Constraint.
%                           The solution must obey || x || <= normConst.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
% Output:
%   - vX                -   Solution Vector.
%                           The optimal solution of the problem 0.5 || A x
%                           - b || ^ 2 subject to || x || <= normConst.
%                           Structure: Vector (Column Vector).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References
%   1.  C
% Remarks:
%   1.  T
% TODO:
%   1.  U
% Release Notes:
%   -   1.0.000     21/08/2017
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

paramLambdaInit = 0;

vX = pinv(mA) * vB;
mI = eye(size(vX, 1));

mAA = mA.' * mA;
mAb = mA.' * vB;

if(norm(vX, 2) <= normConst)
    return;
end

hObjFun = @(paramLambda) norm((mAA + (paramLambda * mI)) \ mAb, 2) - 1;

paramLambda = fzero(hObjFun, paramLambdaInit);

vX = (mAA + (paramLambda * mI)) \ mAb;


end

