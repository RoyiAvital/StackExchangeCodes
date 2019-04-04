function [ mO ] = ImageFilteringFrequencyDomain( mI, mH, paddingMode )
% ----------------------------------------------------------------------------------------------- %
% [ mO ] = ImageFilteringFrequencyDomain( mI, mH, paddingMode )
% Applies Image Filtering in the Frequency Domain.
% Input:
%   - mI                -   Input Image.
%                           Structure: Matrix.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - mH                -   Filtering Kernel.
%                           Structure: Matrix.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - paddingMode       -   Padding Mode.
%                           Sets whether the padding is by Zeros,
%                           Symmetric, Replicate or Circular mode.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, 4}.
% Output:
%   - mI                -   Output Image.
%                           Structure: Matrix.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  MATLAB's 'imfilter()' - https://www.mathworks.com/help/images/ref/imfilter.html.
% Remarks:
%   1.  A
% TODO:
%   1.  
%   Release Notes:
%   -   1.0.000     04/04/2019  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

PADDING_MODE_ZEROS      = 1;
PADDING_MODE_SYMMETRIC  = 2;
PADDING_MODE_REPLICATE  = 3;
PADDING_MODE_CIRCULAR   = 4;

numRows     = size(mI, 1);
numCols     = size(mI, 2);

numRowsKernel = size(mH, 1);
numColsKernel = size(mH, 2);

radiusRows = floor(numRowsKernel / 2);
radiusCols = floor(numColsKernel / 2);

vPadRadius = [radiusRows; radiusCols];

switch(paddingMode)
    case(PADDING_MODE_ZEROS)
        % Size of the Linear Convolution Support
        numRowsL = numRows + numRowsKernel - 1;
        numColsL = numCols + numColsKernel - 1;
        
        % Equivalent of Full Linear Convolution (Zero Padding Built In).
        % This is due the fact padding with zeors at the end with symmetric
        % assumption and shifting yield correct result.
        mO = ifft2(fft2(mI, numRowsL, numColsL) .* fft2(mH, numRowsL, numColsL), 'symmetric');
        
        firstRowIDx = ceil((numRowsKernel + 1) / 2);
        firstColIDx = ceil((numColsKernel + 1) / 2);
        
        mO = mO(firstRowIDx:(firstRowIDx + numRows - 1), firstColIDx:(firstColIDx + numCols - 1));
    case({PADDING_MODE_SYMMETRIC, PADDING_MODE_REPLICATE})
        % Padding to apply "Linear Convolution" on the image pixels.
        mI = PadArray2D(mI, vPadRadius, paddingMode);
        
        numRowsPad = numRows + (2 * radiusRows);
        numColsPad = numCols + (2 * radiusCols);
        
        mHC = mH;
        mHC(numRowsPad, numColsPad) = 0;
        mHC = circshift(mHC, [-radiusRows, -radiusCols]);
        
        mO = ifft2(fft2(mI) .* fft2(mHC), 'symmetric');
        mO = mO((radiusRows + 1):(radiusRows + numRows), (radiusCols + 1):(radiusCols + numCols));
    case(PADDING_MODE_CIRCULAR)
        mHC = mH;
        mHC(numRows, numCols) = 0;
        mHC = circshift(mHC, [-radiusRows, -radiusCols]);
        
        mO = ifft2(fft2(mI) .* fft2(mHC), 'symmetric');
end


end

