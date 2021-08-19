function [ vX, mP ] = ApplyKalmanFilterIteration( vX, mP, vZ, hF, hH, hMf, hMh, mQ, mR )
% ----------------------------------------------------------------------------------------------- %
% [ vX, mP ] = ApplyKalmanFilterIteration( vX, mP, vZ, hF, hH, mQ, mR, mF, mH )
%   Applies iteration of the Kalman Filter (Prediction + Update). Supports
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
%   - mR            -   Measurement Noise Covariance.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - hMf           -   Model Matrix Function.
%                       Generates the matrix `mF` from the state vector vX.
%                       Structure: Function Handler.
%                       Type: Function Handler.
%                       Range: NA.
%   - hMh           -   Measurement Matrix Function.
%                       Generates the matrix `mH` from the state vector vX.
%                       The state vector is hF(vX).
%                       Structure: Function Handler.
%                       Type: Function Handler.
%                       Range: NA.
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
%   1.  If the Complex Mode is selected the function must return complex
%       values in order to work. For instance, if the input function is
%       'norm(vX)' use 'sum(vX .^ 2)' and if the input function is
%       'sum(abs(vX))' use 'sum(sqrt(vX .^ 2))'.
%   2.  This implementation use the Joseph Form of the Covariance Update.
%       While it requires more computational effort it is more stable and
%       guaranteed to generate PSD matrix.
% TODO:
%   1.  U.
% Release Notes:
%   -   1.1.000     25/07/2021  Royi Avital
%       *   Added support for a fnction handle to calculate the matrices mF
%           and mH.
%       *   Add `arguments` block.
%       *   Added a step to ensure symmetry of mP.
%   -   1.0.000     22/08/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    vX (:, 1) {mustBeNumeric, mustBeReal}
    mP (:, :) {mustBeNumeric, mustBeReal, mustBePdMatrix}
    vZ (:, 1) {mustBeNumeric, mustBeReal}
    hF (1, 1) {mustBeFunctionHandler}
    hH (1, 1) {mustBeFunctionHandler}
    hMf (1, 1) {mustBeFunctionHandler}
    hMh (1, 1) {mustBeFunctionHandler}
    mQ (:, :) {mustBeNumeric, mustBeReal, mustBePdMatrix}
    mR (:, :) {mustBeNumeric, mustBeReal, mustBePdMatrix}
end

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

mI = eye(size(vX, 1));

% Prediction Step
vX = hF(vX);
mF = hMf(vX);
mP = (mF * mP * mF.') + mQ;

mH = hMh(vX);

% Update Step
vY = vZ - hH(vX);
mS = (mH * mP * mH.') + mR;
mK = (mP * mH.') / mS;
vX = vX + (mK * vY);
% Joseph Form of the Covariance Update (Numerically more stable)
mT = mI - (mK * mH);
mP = (mT * mP * mT.') + (mK * mR * mK.'); %<! Joseph Form
mP = (mP.' + mP) / 2; %<! Ensure symmetry


end


function [ ] = mustBePdMatrix( mX )
% https://www.mathworks.com/matlabcentral/answers/107552
if(issymmetric(mX) && (all(eig(mX) > 0)))
    
else
    errorId     = 'mustBePdMatrix:notPositiveDefiniteMatrix';
    errorMsg    = 'The input must be a Positive Definite Matrix';
    throwAsCaller(MException(errorId, errorMsg));
end


end


function [ ] = mustBeFunctionHandler( hF )
% https://www.mathworks.com/matlabcentral/answers/107552
if(~isa(hF, 'function_handle'))
    errorId     = 'mustBeFunctionHandler:notFunctionHandler';
    errorMsg    = 'The input must be a Function Handler';
    throwAsCaller(MException(errorId, errorMsg));
end


end

