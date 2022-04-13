function [ vW, vV ] = LmsFilterAccel( vW, vV, vY, vD, paramL, numSamples, stepSizeLms, stepSizeAccel, normalizeMode )
% ----------------------------------------------------------------------------------------------- %
% [ vW ] = LmsFilter( vY, vD, stepSize, vW, paramN, paramM, paramL, normalizeMode )
% Applies the Least Mean Squares Adaptive Filter for optimal 'vW' weights
% given the reference signal vD. It is accelerated (The convergence rate)
% by Nesterov Method.
% Input:
%   - vW                -   Filter Taps.
%                           The adaptive filter taps (To be updated).
%                           Structure: Vector (paramL x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vV                -   Nesterov Vector.
%                           Vector of momentum vectors. Should be
%                           initialized to zeros or the value of the output
%                           of previous session.
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
%   - stepSizeLms       -   Step Size
%                           The step size (paramMu) of the LMS filter.
%                           Basically the SGD step size.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - stepSizeAccel     -   Acceleration step Size.
%                           The step size of the Nesterov Acceleration
%                           step.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: [0, 1).
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
%   - vV                -   Momentum Vector.
%                           Vector of momentum vectors.
%                           Structure: Vector (paramL x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     07/08/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

DELTA_PARAM = 1e-5;

vG          = zeros(paramL, 1);

% paramL %<! Order of the filter

for ii = paramL:numSamples
    vYn     = vY(ii:-1:(ii - paramL + 1));
    dSample = vD(ii);
    
    % Nesterov Step (Maybe use the FISTA weighing?)
    zSample = (vW + (stepSizeAccel * vV)).' * vYn;
    
    eSample = dSample - zSample;
    
    if(normalizeMode == ON)
        vG(:) = (1 / ((vYn.' * vYn) + DELTA_PARAM)) * eSample * vYn;
    else
        vG(:) = eSample * vYn;
    end
    
    vV(:) = (stepSizeAccel * vV) + (stepSizeLms * vG);
    vW(:) = vW + vV;
    
end


end

