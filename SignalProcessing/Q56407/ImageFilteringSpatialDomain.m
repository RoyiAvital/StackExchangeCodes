function [ mO ] = ImageFilteringSpatialDomain( mI, mH, paddingMode )
% ----------------------------------------------------------------------------------------------- %
% [ mO ] = ImageFilteringSpatialDomain( mI, mH, paddingMode )
% Applies Image Filtering in the Spatial Domain.
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

numRowsKernel = size(mH, 1);
numColsKernel = size(mH, 2);

filterRadiusV = floor(size(mH, 1) / 2);
filterRadiusH = floor(size(mH, 2) / 2);

mI = PadArray2D(mI, [filterRadiusV, filterRadiusH], paddingMode);

mO = conv2(mI, mH, 'valid');

if(mod(numRowsKernel, 2) == 0)
    mO = mO(2:end, :);
end
if(mod(numColsKernel, 2) == 0)
    mO = mO(:, 2:end);
end


end

