function [ vO ] = ConvolutionDft( vS, vK, convShape )
% ----------------------------------------------------------------------------------------------- %
% [ vO ] = ConvolutionDft( vS, vK, convShape )
% Applying 1D Linear Convolution using the DFT.
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

signalLength = size(vS, 1);
kernelLength = size(vK, 1);

numSamlpesOutput = signalLength + kernelLength - 1; %<! Linear Convolution, Full

vO = ifft(fft(vS, numSamlpesOutput) .* fft(vK, numSamlpesOutput), 'symmetric');

switch(convShape)
    case(CONVOLUTION_SHAPE_SAME)
        idxFirst    = 1 + floor(kernelLength / 2);
        idxLast     = idxFirst + signalLength - 1;
        vO          = vO(idxFirst:idxLast);
    case(CONVOLUTION_SHAPE_VALID)
        idxFirst    = kernelLength;
        idxLast     = (signalLength + kernelLength - 1) - kernelLength + 1;
        vO          = vO(idxFirst:idxLast);
end


end

