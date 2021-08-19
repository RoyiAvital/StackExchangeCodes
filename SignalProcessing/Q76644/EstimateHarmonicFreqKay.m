function [ estFreq ] = EstimateHarmonicFreqKay( vX, samplingFreq, estType )
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

arguments
    vX (:, 1) {mustBeNumeric}
    samplingFreq (1, 1) {mustBeNumeric, mustBeReal, mustBePositive}
    estType (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger, mustBeMember(estType, [1, 2])} = 2
end

EST_TYPE_1 = 1;
EST_TYPE_2 = 2; %<! Better in high SNR

numSamples = size(vX, 1);

estFreq = 0;
switch(estType)
    case(EST_TYPE_1)
        for ii = 1:(numSamples - 1)
            estFreq = estFreq + angle(vX(ii)' * vX(ii + 1));
        end
        estFreq = estFreq / (2 * pi * (numSamples - 1));
    case(EST_TYPE_2)
        weightDen = (numSamples ^ 3) - numSamples;
        for ii = 1:(numSamples - 1)
            sampleWeight    = (6 * ii * (numSamples - ii)) / weightDen;
            estFreq         = estFreq + (sampleWeight * angle(vX(ii)' * vX(ii + 1)));
        end
        estFreq = estFreq / (2 * pi);
end

estFreq = samplingFreq * estFreq; %<! Moving from normalized frequency to absolute frequency


end

