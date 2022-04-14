function [ vW ] = LmsFilter( vW, vY, vD, paramL, numSamples, stepSize, normalizeMode )
% ----------------------------------------------------------------------------------------------- %
% [ vW ] = LmsFilter( vY, vD, stepSize, vW, paramN, paramM, paramL, normalizeMode )
% Applies the Least Mean Squares Adaptive Filter for optimal 'vW' weights
% given the reference signal vD.
% Input:
%   - vW                -   Filter Taps.
%                           The adaptive filter taps (To be updated).
%                           Structure: Vector (paramL x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vY                -   Input Signal.
%                           The signal to be filtered to by the adaptive
%                           filter to match the reference signal.
%                           Structure: Vector (numSamples x 1.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vD                -   Reference Signal.
%                           The reference signal
%                           Structure: Vector (numSamples x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - paramL            -   Number of Taps.
%                           Number of taps of the adaptive filter.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, ...}.
%   - numSamples        -   Number of Samples.
%                           Number of samples of the input vector 'vY' and
%                           'vD'.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, ..., min(length(vY), length(vD)}.
%   - stepSize          -   Step Size
%                           The step size (paramMu) of the LMS filter.
%                           Basically the SGD step size.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - normalizeMode     -   Normalize Mode.
%                           Normalizes the step size by the norm of the
%                           samples.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1} / {OFF, ON}.
% Output:
%   - vW                -   Filter Taps.
%                           The updated adaptive filter taps.
%                           Structure: Vector (paramL x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  A
% Remarks:
%   1.  The values of the step size must be in the range (0, 2 / lambdaMax)
%       where lambdaMax is the maximum eigen value of the samples
%       covariance.
%   2.  In case of normalization it is usually will converge for stepSize
%       within the range [0, 2).
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     07/08/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

% Predictor variant

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

DELTA_PARAM = 1e-5;

% paramL %<! Order of the filter

for ii = paramL:numSamples
    vYn     = vY(ii:-1:(ii - paramL + 1));
    dSample = vD(ii);
    
    zSample = vW.' * vYn;
    
    eSample = dSample - zSample;
    
    if(normalizeMode == ON)
        vW(:) = vW + (stepSize * eSample * vYn);
    else
        vW(:) = vW + ((stepSize / ((vYn.' * vYn) + DELTA_PARAM)) * eSample * vYn);
    end
    
end


end

