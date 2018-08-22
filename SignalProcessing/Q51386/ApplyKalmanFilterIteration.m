function [ vX, mP ] = ApplyKalmanFilterIteration( vX, mP, vZ, hF, hH, mQ, mR, mF, mH )
% ----------------------------------------------------------------------------------------------- %
% [ vX, mP ] = ApplyKalmanFilterIteration( vX, mP, vZ, hF, hH, mQ, mR, mF, mH )
%   Applies iteration of the Kalman Filter (Predicition + Update). Supports
%   both Linear and Non Linear Mode (Extended Kalman Filter).
% Input:
%   - vX            -   Input State Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mP            -   Error Covariance.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vZ            -   Measurement Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - hF            -   Model Function.
%                       Structure: Function Handler.
%                       Type: Function Handler.
%                       Range: NA.
%   - hH            -   Measurement Function.
%                       Structure: Function Handler.
%                       Type: Function Handler.
%                       Range: NA.
%   - mQ            -   Process Noise Covariance.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mR            -   Measuremetn Noise Covariance.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mF            -   Model Function Jacobian at vX.
%                       Optional parameter.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mH            -   Measuremetn Function Jacobian at hF(vX).
%                       Optional parameter.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - vX            -   Updated State Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mP            -   Error Covariance.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  Extended Kalman Filter (Wikipedia) - https://en.wikipedia.org/wiki/Extended_Kalman_filter.
%   2.  Kalman Filter (Wikipedia) - https://en.wikipedia.org/wiki/Kalman_filter.
% Remarks:
% Remarks:
%   1.  If the Complex Mode is selected the function must return complex
%       values in order to work. For instance, if the input function is
%       'norm(vX)' use 'sum(vX .^ 2)' and if the input function is
%       'sum(abs(vX))' use 'sum(sqrt(vX .^ 2))'.
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     22/08/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;
DIFF_MODE_COMPLEX   = 4;

diffMode    = DIFF_MODE_FORWARD;
epsVal      = 1e-8;

if(~exist('mF', 'var'))
    mF = CalcFunJacob(vX, hF, diffMode, epsVal);
end

if(~exist('mH', 'var'))
    mH = CalcFunJacob(hF(vX), hH, diffMode, epsVal);
end

mI = eye(size(vX, 1));

% Prediction Step
vX = hF(vX);
mP = (mF * mP * mF.') + mQ;

% Update Step

vY = vZ - hH(vX);
mS = (mH * mP * mH.') + mR;
mK = (mP * mH.') / mS;
vX = vX + (mK * vY);
mT = mI - (mK * mH);
mP = (mT * mP * mT.') + (mK * mR * mK.'); %<! Jospeh Form


end

