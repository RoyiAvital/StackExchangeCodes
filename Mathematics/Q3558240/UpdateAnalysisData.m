function [ mObjFunVal, mSolErrNorm ] = UpdateAnalysisData( mObjFunVal, mSolErrNorm, mX, hObjFun, sCvxSol, solverIdx )
% ----------------------------------------------------------------------------------------------- %
% Remarks:
%   1.  T
% Known Issues:
%   1.  A
% TODO:
%   1.  A
% Release Notes:
%   -   1.0.000     23/11/2016
%       *   First realease version.
% ----------------------------------------------------------------------------------------------- %

cvxOptVal = sCvxSol.cvxOptVal;
vXCvx = sCvxSol.vXCvx;

numIterations = size(mSolErrNorm, 1);

for ii = 1:numIterations

    mObjFunVal(ii, solverIdx)   = abs(hObjFun(mX(:, ii)) - cvxOptVal);
    mSolErrNorm(ii, solverIdx)  = norm(mX(:, ii) - vXCvx);

end


end

