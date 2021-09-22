function [ vS ] = RecSignalFreqSamples( vF, mF, vFIdx )
% ----------------------------------------------------------------------------------------------- %
%[ estFreq ] = EstimateSineFreqKay( vX, samplingFreq, estType )
% Estimates the frequency of a single Real Harmonic signal with arbitrary
% phase.
% Input:
%   - vX                -   Input Samples.
%                           The vector to be optimized. Initialization of
%                           the iterative process.
%                           Structure: Vector (numSamples X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - paramLambda       -   Parameter Lambda.
%                           The L1 Regularization parameter.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - numIterations     -   Number of Iterations.
%                           Number of iterations of the algorithm.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range {1, 2, ...}.
% Output:
%   - estFreq           -   Number of Iterations.
%                           Number of iterations of the algorithm.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range {1, 2, ...}.
% References
%   1.  Steven Kay - A Fast and Accurate Single Frequency Estimator.
% Remarks:
%   1.  It would work with complex numbers (Harmonic Signla) by changing:
%       vX(ii) * vX(ii + 1) -> vX(ii)' * vX(ii + 1).
%   2.  fds
% Known Issues:
%   1.  C
% TODO:
%   1.  D
% Release Notes:
%   -   1.0.000     09/08/2021  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

% arguments
%     vX (:, 1) {mustBeNumeric, mustBeReal}
%     samplingFreq (1, 1) {mustBeNumeric, mustBeReal, mustBePositive}
%     estType (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger, mustBeMember(estType, [1, 2])} = 2
% end

numSamples = size(mF, 2);

mFR = real(mF);
mFI = imag(mF);

vS = real(SolveBasisPursuitLp001([mFR(vFIdx, :); mFI(vFIdx, :)], [real(vF); imag(vF)]));
% vSRec = real(SolveBasisPursuitLp002(mF(vFIdx, :), vF));

% vS = complex(vSRec(1:numSamples), vSRec((numSamples + 1):end));


end

