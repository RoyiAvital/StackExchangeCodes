function [ vO ] = ConvolutionOverlapSave( vS, vK, convShape )
% ----------------------------------------------------------------------------------------------- %
% [ vO ] = ConvolutionDft( vS, vK, convShape )
% Applying 1D Linear Convolution using the Overlap and Save approach. The
% function calculates the optimal DFT Window.
% Input:
%   - vS                -   Input 1D Convolution Signal.
%                           Structure: Vector (signalLength, 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vK                -   Input 1D Convolution Kernel.
%                           Structure: Vector (kernelLength, 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - convShape         -   Convolution Shape.
%                           The shape of the convolution which the output
%                           convolution matrix should represent. The
%                           options should match MATLAB's conv() function
%                           - Full / Same / Valid.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3}.
% Output:
%   - vS                -   Input 1D Convolution Output Vector.
%                           Structure: Vector (outputLength, 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  MATLAB's 'conv()' - https://www.mathworks.com/help/matlab/ref/conv.html.
%   2.  Overlap and Save Method (Wikipedia) - https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method.
% Remarks:
%   1.  The signals must be columns signals.
%   2.  It is assumed that the input signal is not shorter than the kernel.
% TODO:
%   1.  C
%   Release Notes:
%   -   1.0.000     27/04/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;

signalLength    = size(vS, 1); %<! K
kernelLength    = size(vK, 1); %<! M
dftLength       = CalcOptimalDftLength(signalLength, kernelLength); %<! N
convWinLength   = dftLength - kernelLength + 1; %<! L

paddSignalLength = ceil((signalLength + kernelLength - 1) / convWinLength) * dftLength;

vSS = [zeros(kernelLength - 1, 1); vS; zeros(paddSignalLength - kernelLength - 1 + signalLength, 1)];

numSteps    = ceil((signalLength + kernelLength - 1) / convWinLength);
vKD         = fft(vK, dftLength);
vO          = zeros(numSteps * convWinLength, 1);
vOO         = zeros(dftLength, 1);
idxPos      = 0;
for ii = 1:numSteps
    firstIdx = idxPos + 1;
    lastIdx = idxPos + dftLength;
    vOO(:) = ifft(fft(vSS(firstIdx:lastIdx)) .* vKD, 'symmetric');
    lastIdx = idxPos + convWinLength;
    vO(firstIdx:lastIdx) = vOO(kernelLength:dftLength);
    idxPos = idxPos + convWinLength;
end

switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        idxFirst    = 1;
        idxLast     = signalLength + kernelLength - 1;
    case(CONVOLUTION_SHAPE_SAME)
        idxFirst    = 1 + floor(kernelLength / 2);
        idxLast     = idxFirst + signalLength - 1;
    case(CONVOLUTION_SHAPE_VALID)
        idxFirst    = kernelLength;
        idxLast     = (signalLength + kernelLength - 1) - kernelLength + 1;
end

vO          = vO(idxFirst:idxLast);


end


function [ dftLength ] = CalcOptimalDftLength( signelLength, kernelLength )

oConvOs = @(dftLength) (dftLength * log2(dftLength) + dftLength) / (dftLength - kernelLength + 1);

outputLength = signelLength + kernelLength;

firstPow2   = ceil(log2(kernelLength));
lastPow2    = ceil(log2(outputLength));

pow2 = firstPow2;
optNumOps = oConvOs(2 ^ pow2);

for pow2 = (firstPow2 + 1):lastPow2
    currNumOps = oConvOs(2 ^ pow2); 
    if(currNumOps > optNumOps)
        break;
    end
    optNumOps = currNumOps;
end

dftLength = 2 ^ pow2;


end

