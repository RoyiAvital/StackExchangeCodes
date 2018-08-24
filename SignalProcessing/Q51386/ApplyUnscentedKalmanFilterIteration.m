function [ vX, mP ] = ApplyUnscentedKalmanFilterIteration( vX, mP, vZ, hF, hH, mQ, mR )
% ----------------------------------------------------------------------------------------------- %
% [ vX, mP ] = ApplyUnscentedKalmanFilterIteration( vX, mP, vZ, hF, hH, mQ, mR )
%   Applies iteration of the Unscented Kalman Filter (Predicition +
%   Update). Supports both Linear and Non Linear Modes.
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
%   1.  Unscented Kalman Filter (Wikipedia) - https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter.
%   2.  Unscented Tranform (Wikipedia) - https://en.wikipedia.org/wiki/Unscented_transform.
%   3.  Kalman Filter (Wikipedia) - https://en.wikipedia.org/wiki/Kalman_filter.
%   4.  Lecture 5: Unscented Kalman Filter and General Gaussian Filtering (Simo Sarkka).
% Remarks:
%   1.  I
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     24/08/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

paramAlpha  = 1e-3;
paramBeta   = 2;
paramKappa  = 0;

stateOrder  = size(vX, 1);
measOrder   = size(vZ, 1);

% Pre Process
[vWm, vWc, scalingFctr] = CalcSigmaPointsWeights(paramAlpha, paramBeta, paramKappa, stateOrder);

% Prediction Step
mX = GenerateSigmaPts(vX, mP, scalingFctr);
[vX, mX, mP, mXx] = ApplyUnscentedTransform(hF, mX, vWm, vWc, stateOrder);
mP = mP + mQ; %<! Adding the Process Covariance (Additive as assumed to be independent)

% Update Step
[vY, ~, mS, mYy] = ApplyUnscentedTransform(hH, mX, vWm, vWc, measOrder);
mS = mS + mR; %<! Adding the Measurement Covariance (Additive as assumed to be independent)

mC = mXx * diag(vWc) * mYy.';
mK = mC / mS;
vX = vX + (mK * (vZ - vY));
mP = mP - (mK * mS * mK.');
mP = 0.5 * (mP + mP.'); %<! Symmetrizing


end

