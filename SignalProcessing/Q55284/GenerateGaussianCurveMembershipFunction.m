function [ vP ] = GenerateGaussianCurveMembershipFunction( vX, meanVal, stdVal )
% ----------------------------------------------------------------------------------------------- %
% [ vF ] = AppltDft( vX, numFreqSamples )
%   Applt the N Point ('numFreqSamples') DFT Trnasform on input.
% Input:
%   - vX            -   Grid Support.
%                       Support of the Gaussian Funtion.
%                       Structure: Vector.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - meanVal       -   Mean Value (Shift).
%                       Sets the Mean Value (Shifting) of the Gaussian
%                       Function.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - meanVal       -   Mean Value (Shift).
%                       Sets the Mean Value (Shifting) of the Gaussian
%                       Function.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - vP            -   Output Vector.
%                       The function value over the support.
%                       Structure: Vector.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  Gaussian Curve Membership Function (MATLAB Documentation) - https://www.mathworks.com/help/fuzzy/gaussmf.html.
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     07/02/2019  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

vP = exp(-((vX - meanVal) .^ 2) / (2 * stdVal * stdVal));


end

