function [ mK ] = CreateConvMtxSparse( vK, numElements, convShape )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;

kernelLength    = length(vK);

switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        rowIdxFirst = 1;
        rowIdxLast  = numElements + kernelLength - 1;
        outputSize  = numElements + kernelLength - 1;
    case(CONVOLUTION_SHAPE_SAME)
        rowIdxFirst = 1 + floor(kernelLength / 2);
        rowIdxLast  = rowIdxFirst + numElements - 1;
        outputSize  = numElements;
    case(CONVOLUTION_SHAPE_VALID)
        rowIdxFirst = kernelLength;
        rowIdxLast  = (numElements + kernelLength - 1) - kernelLength + 1;
        outputSize  = numElements - kernelLength + 1;
end

mtxIdx = 0;

% The sparse matrix constructor ignores valus of zero yet the Row / Column
% indices must be valid indices (Positive integers). Hence 'vI' and 'vJ'
% are initialized to 1 yet for invalid indices 'vV' will be 0 hence it has
% no effect.
vI = ones(numElements * kernelLength, 1);
vJ = ones(numElements * kernelLength, 1);
vV = zeros(numElements * kernelLength, 1);

for jj = 1:numElements
    for ii = 1:kernelLength
        if((ii + jj - 1 >= rowIdxFirst) && (ii + jj - 1 <= rowIdxLast))
            % Valid otuput matrix row index
            mtxIdx = mtxIdx + 1;
            vI(mtxIdx) = ii + jj - rowIdxFirst;
            vJ(mtxIdx) = jj;
            vV(mtxIdx) = vK(ii);
        end
    end
end

mK = sparse(vI, vJ, vV, outputSize, numElements);


end

