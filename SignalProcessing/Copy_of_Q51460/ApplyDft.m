function [ vF ] = ApplyDft( vX, numFreqSamples )
% ----------------------------------------------------------------------------------------------- %
% [ vF ] = AppltDft( vX, numFreqSamples )
%   Applt the N Point ('numFreqSamples') DFT Trnasform on input.
% Input:
%   - vX            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - numFreqSamples-   Number of Frequency Samples.
%                       Sets the number of Smaples in the Frequency Domain.
%                       Must by not smaller than the size of teh input.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, 3, 4}.
% Output:
%   - vF            -   Output Vector.
%                       The 'numFreqSamples' point DFT of 'vX'.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  Discrete Fourier Transform (Wikipedia) - https://en.wikipedia.org/wiki/Discrete_Fourier_transform.
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     26/08/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numSamples = size(vX, 1);

vF = complex(zeros(numFreqSamples, 1));

if(numFreqSamples > numSamples)
    vX(numFreqSamples) = 0;
end

complexFactor = (-2i * pi) / numFreqSamples;

for kk = 1:numFreqSamples
    for nn = 1:numFreqSamples
        vF(kk) = vF(kk) + vX(nn) * exp(complexFactor * (nn - 1) * (kk - 1));
    end
end


end

