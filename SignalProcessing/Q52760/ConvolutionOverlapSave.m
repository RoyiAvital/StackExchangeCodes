function [ vO ] = ConvolutionOverlapSave( vS, vK, convShape )
% ----------------------------------------------------------------------------------------------- %
% [ mK ] = CreateConvMtx1D( vK, numElements, convShape )
% Generates a Convolution Matrix for 1D Kernel (The Vector vK) with
% support for different convolution shapes (Full / Same / Valid). The
% matrix is build such that for a signal 'vS' with 'numElements = size(vS
% ,1)' the following are equiavlent: 'mK * vS' and conv(vS, vK,
% convShapeString);
% Input:
%   - vK                -   Input 1D Convolution Kernel.
%                           Structure: Vector.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - numElements       -   Number of Elements.
%                           Number of elements of the vector to be
%                           convolved with the matrix. Basically set the
%                           number of columns of the Convolution Matrix.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, ...}.
%   - convShape         -   Convolution Shape.
%                           The shape of the convolution which the output
%                           convolution matrix should represent. The
%                           options should match MATLAB's conv2() function
%                           - Full / Same / Valid.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3}.
% Output:
%   - mK                -   Convolution Matrix.
%                           The output convolution matrix. The product of
%                           'mK' and a vector 'vS' ('mK * vS') is the
%                           convolution between 'vK' and 'vS' with the
%                           corresponding convolution shape.
%                           Structure: Matrix (Sparse).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  MATLAB's 'convmtx()' - https://www.mathworks.com/help/signal/ref/convmtx.html.
% Remarks:
%   1.  The output matrix is sparse data type in order to make the
%       multiplication by vectors to more efficient.
%   2.  In caes the same convolution is applied on many vectors, stacking
%       them into a matrix (Each signal as a vector) and applying
%       convolution on each column by matrix multiplication might be more
%       efficient than applying classic convolution per column.
% TODO:
%   1.  
%   Release Notes:
%   -   1.0.000     20/01/2019  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;

% numSamplesSignal = size(vS, 1);
% numSamplesKernel = size(vK, 1);
% 
% numSamlpesOutput = numSamplesSignal + numSamplesKernel - 1; %<! Linear Convolution, Full
% 
% overlapLength = numSamplesKernel - 1;
% % fftLength = CalcOptimalStepSize(numSamplesSignal, numSamplesKernel);
% fftLength = numSamplesKernel + 3;
% 
% vSS = [vS; zeros(mod(-numSamplesSignal, overlapLength), 1); zeros(overlapLength, 1)];


K = size(vS, 1);
M = size(vK, 1); %<! Filter Length
N = CalcOptimalDftLength(K, M);
L = N - M + 1;

P = ceil((K + M - 1) / L) * N;

vSS = [zeros(M - 1, 1); vS; zeros(P - M - 1 + K, 1)];

numSteps    = ceil((K + M - 1) / L);
vKD         = fft(vK, N);
vO          = zeros(numSteps * L, 1);
vOO         = zeros(N, 1);
idxPos      = 0;
for ii = 1:numSteps
    firstIdx = idxPos + 1;
    lastIdx = idxPos + N;
    vOO(:) = ifft(fft(vSS(firstIdx:lastIdx)) .* vKD, 'symmetric');
    lastIdx = idxPos + L;
    vO(firstIdx:lastIdx) = vOO(M:N);
    idxPos = idxPos + L;
end



switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        idxFirst    = 1;
        idxLast     = K + M - 1;
    case(CONVOLUTION_SHAPE_SAME)
        idxFirst    = 1 + floor(M / 2);
        idxLast     = idxFirst + K - 1;
    case(CONVOLUTION_SHAPE_VALID)
        idxFirst    = M;
        idxLast     = (K + M - 1) - M + 1;
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

