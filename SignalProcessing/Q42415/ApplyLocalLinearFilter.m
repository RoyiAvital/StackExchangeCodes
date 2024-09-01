function [ mOutputImage ] = ApplyLocalLinearFilter( mInputImage, filterRadius, regFctr )
% ----------------------------------------------------------------------------------------------- %
%[ mOutputImage ] = ApplyLocalLinearFilter( mInputImage, filterRadius, regFctr )
% Applying Linear Edge Preserving Smoothing Filter based on Local Linear (Affine) model.
% Input:
%   - inputImage    -   Input Image.
%                       Structure: Image Matrix (Single Channel).
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
%   - filterRadius  -   Filter Radius.
%                       The filter radius.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ...}.
%   - regFctr       -   Regularization Factor.
%                       Regularize the local covariance (Variance).
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
% Output:
%   - mOutputImage  -   Output Image.
%                       Structure: Image Matrix (Single Channel).
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
% References:
%   1.  "Guided Image Filtering".
% Remarks:
%   1.  This is basically estimating the Linear Function (Affine)
%       parameters for the Local Window. Namely, the output is a Linear
%       combination of the input Window and a DC Factor. The final step is
%       aggregation (Uniform) off all estimations of the parameters.
%   2.  Prefixes:
%       -   'v' - Vector.
%       -   'm' - Matrix.
%       -   't' - Tensor (Multi Dimension Matrix)
%       -   's' - Struct.
%       -   'c' - Cell Array.
%   3.  The calculation of the Local Variance might be negative due to
%       numerical difficulties. If artifacts appear, this might be the
%       cause. Usually using matrices of type 'double' solves it.
%   4.  This implementation is `ApplyGuidedFilter` where the Guiding Image
%       is the Input Image.
%   5.  Speed optimization can be achieved by wiser use of 'mNumEffPixels'.
%       Instead of dividing by it calculate its reciprocal once. Moreover,
%       it can be used only once in the aggregation process.
% TODO:
%   1.  Create Multi Variable Linear Model.
%   2.  Some speed optimization could be made (Taking advantage of 'mNumEffPixels').
% Release Notes:
%   -   1.0.000     05/01/2016  Royi Avital
%       *   First release version
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF = 0;
ON  = 1;

BORDER_TYPE_CONSTANT    = 1;
BORDER_TYPE_CIRCULAR    = 2;
BORDER_TYPE_REPLICATE   = 3;
BORDER_TYPE_SYMMETRIC   = 4;

numRows = size(mInputImage, 1);
numCols = size(mInputImage, 2);

borderType      = BORDER_TYPE_CONSTANT;
borderValue     = 0;
normalizeFlag   = OFF;

mNumEffPixels = ApplyBoxFilter(ones([numRows, numCols]), filterRadius, borderType, borderValue, normalizeFlag);

mLocalMean          = ApplyBoxFilter(mInputImage, filterRadius, borderType, borderValue, normalizeFlag) ./ mNumEffPixels;
mLocalMeanSquare    = ApplyBoxFilter((mInputImage .* mInputImage), filterRadius, borderType, borderValue, normalizeFlag) ./ mNumEffPixels;
mLocalCovariance    = mLocalMeanSquare - (mLocalMean .* mLocalMean);
% This step is needed only for cases the calculation is using Integral
% Images for the calculation of the Box Blur.
% mLocalCovariance    = max(mLocalCovariance, 0);

% Similar to "Soft Thresholding" of the Covariance
mACoef = mLocalCovariance ./ (mLocalCovariance + regFctr);
mBCoef = mLocalMean - (mACoef .* mLocalMean);

% The local mean of each coefficient (Uniform aggregation)
mACoef = ApplyBoxFilter(mACoef, filterRadius, borderType, borderValue, normalizeFlag) ./ mNumEffPixels;
mBCoef = ApplyBoxFilter(mBCoef, filterRadius, borderType, borderValue, normalizeFlag) ./ mNumEffPixels;

mOutputImage = (mACoef .* mInputImage) + mBCoef;


end

