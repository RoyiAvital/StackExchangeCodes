function [ vTheta ] = UpdateSequentialLsModel( mH, inputSample, vTheta )
% ----------------------------------------------------------------------------------------------- %
% [ vTheta ] = UpdateSequentialLsModel( mH, inputSample, vTheta )
% Applies update step of the Parameters Vector of Linear Least Squares in a
% sequential form.
% Input:
%   - vK                -   Model Matrix.
%                           The Model Matrix up to the index of the last
%                           sample.
%                           Structure: Matrix.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - inputSample       -   Input Sample.
%                           The last sample in the sequential model.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vTheta            -   Theta.
%                           The previous iteration output of the estimated
%                           parameters vector.
%                           Structure: Vector (Column).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% Output:
%   - vTheta            -   Theta.
%                           The updated output of the estimated parameters
%                           vector. 
%                           Structure: Vector (Column).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  See "Sequential Form of the Least Squares for Linear Least Squares Model" - https://dsp.stackexchange.com/a/56670/128.
% Remarks:
%   1.  A
% TODO:
%   1.  
%   Release Notes:
%   -   1.0.000     13/04/2019  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

vH = mH(end, :).';
mR = pinv(mH(1:end - 1, :).' * mH(1:end - 1, :));

mK = (mR * vH) / (1 + (vH.' * mR * vH));

vTheta = vTheta + (mK * (inputSample - (vH.' * vTheta)));


end

