function [ vX, paramLambda ] = SolveLsNormSquaredConst( mA, vB, normSquaredConst, numIterations )
% ----------------------------------------------------------------------------------------------- %
%[ vX ] = SolveLsNormConst( mA, vB, normConst )
% Solves norm constrained Least Squares problem by finding the optimal Dual
% Variable (paramLambda) of the KKT Conditions by successively solving
% Tikhonov Regularized Least Squares problems.
% The objective Function is given by:
% \arg \min_{x} \frac{1}{2} {\left\| A x - b \right\|}_{2}^{2}
% subject to {\left\| x \right\|}_{2}^{2} \leq normSquaredConst
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
%   1.  See - https://stats.stackexchange.com/questions/401212.
% Remarks:
%   1.  T
% TODO:
%   1.  U
% Release Notes:
%   -   1.0.000     14/04/2019
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

paramLambda = 0; %<! Initialization

vX = pinv(mA) * vB;
mI = eye(size(vX, 1));

mAA = mA.' * mA;
mAb = mA.' * vB;

if((norm(vX, 2) ^ 2) <= normSquaredConst)
    return;
end

hObjFun     = @(paramLambda) (norm((mAA + (2 * paramLambda * mI)) \ mAb, 2) ^ 2) - normSquaredConst;
hObjFunGrad = @(paramLambda) -4 * vB.' * mA * inv(mAA + (2 * paramLambda * mI)) * inv(mAA + (2 * paramLambda * mI)) * inv(mAA + (2 * paramLambda * mI)) * mA.' * vB;

for ii = 1:numIterations
    paramLambda = paramLambda - (hObjFun(paramLambda) / hObjFunGrad(paramLambda));
end


% paramLambda = fzero(hObjFun, 1e-6); %<! Doesn't work for some reason
% paramLambda = FindZero(hObjFun, paramLambda, 1e6);

% hObjFun     = @(paramLambda) (((norm((mAA + (2 * paramLambda * mI)) \ mAb, 2) ^ 2) - normSquaredConst) ^ 2);
% % paramLambda = fminbnd(hObjFun, 0, 2000);
% paramLambda = fminsearch(hObjFun, paramLambda);

vX = (mAA + (2 * paramLambda * mI)) \ mAb;


end

