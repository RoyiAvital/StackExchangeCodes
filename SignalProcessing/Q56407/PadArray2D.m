function [ mO ] = PadArray2D( mI, vPadRadius, paddingMode )
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

PADDING_MODE_ZEROS      = 1;
PADDING_MODE_SYMMETRIC  = 2;
PADDING_MODE_REPLICATE  = 3;
PADDING_MODE_CIRCULAR   = 4;

switch(paddingMode)
    case(PADDING_MODE_ZEROS)
        mO = PadArrayZeros(mI, vPadRadius);
    case(PADDING_MODE_SYMMETRIC)
        mO = PadArraySymmetric(mI, vPadRadius);
    case(PADDING_MODE_REPLICATE)
        mO = PadArrayReplicate(mI, vPadRadius);
    case(PADDING_MODE_CIRCULAR)
        mO = PadArrayCircular(mI, vPadRadius);
end


end


function [ mO ] = PadArrayZeros( mI, vPadRadius )
%PADARRAYREPLICATE Summary of this function goes here
%   Detailed explanation goes here

numRows     = size(mI, 1);
numCols     = size(mI, 2);

dataClass = class(mI);

numRowsOut = numRows + (2 * vPadRadius(1));
numColsOut = numCols + (2 * vPadRadius(2));

mO = zeros(numRowsOut, numColsOut, dataClass);


mO((vPadRadius(1) + 1):(vPadRadius(1) + numRows), (vPadRadius(2) + 1):(vPadRadius(2) + numCols)) = mI;


end


function [ mO ] = PadArraySymmetric( mI, vPadRadius )
%PADARRAYREPLICATE Summary of this function goes here
%   Detailed explanation goes here

numRows     = size(mI, 1);
numCols     = size(mI, 2);

vImageSize = [numRows; numCols];


cPadIdx = cell(1, length(vImageSize));

for ii = 1:length(vImageSize)
    dimSize     = vImageSize(ii);
    padRadius   = vPadRadius(ii); 
    vE1         = uint32(padRadius:-1:1);
    vE2         = uint32(dimSize:-1:(dimSize - padRadius + 1));
    cPadIdx{ii} = [vE1, 1:dimSize, vE2];
end

mO = mI(cPadIdx{:});


end


function [ mO ] = PadArrayReplicate( mI, vPadRadius )
%PADARRAYREPLICATE Summary of this function goes here
%   Detailed explanation goes here

numRows     = size(mI, 1);
numCols     = size(mI, 2);

vImageSize = [numRows; numCols];


cPadIdx = cell(1, length(vImageSize));

for ii = 1:length(vImageSize)
    dimSize     = vImageSize(ii);
    padRadius   = vPadRadius(ii); 
    vE          = uint32(ones(1, padRadius));
    cPadIdx{ii} = [vE, 1:dimSize, (dimSize * vE)];
end

mO = mI(cPadIdx{:});


end


function [ mO ] = PadArrayCircular( mI, vPadRadius )
%PADARRAYREPLICATE Summary of this function goes here
%   Detailed explanation goes here

numRows     = size(mI, 1);
numCols     = size(mI, 2);

vImageSize = [numRows; numCols];

cPadIdx = cell(1, length(vImageSize));

for ii = 1:length(vImageSize)
    dimSize     = vImageSize(ii);
    padRadius   = vPadRadius(ii); 
    vE          = uint32(1:dimSize);
    cPadIdx{ii} = vE(mod(-padRadius:dimSize + padRadius - 1, dimSize) + 1);
end

mO = mI(cPadIdx{:});


end

