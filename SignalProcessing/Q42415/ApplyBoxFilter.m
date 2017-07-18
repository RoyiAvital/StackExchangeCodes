function [ mOutputImage ] = ApplyBoxFilter( mInputImage, boxRadius, borderType, borderValue, normalizeFlag )
% ----------------------------------------------------------------------------------------------- %
% [ mFilteredImage ] = ApplyBoxFilter( mInputImage, boxBlurKernelRadius )
%   Applies Box Filter using Integral Images.
% Input:
%   - mInputImage   -   Input Image.
%                       Structure: Image Matrix (Single Channel)
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
%   - boxRadius     -   Box Radius.
%                       The radius of the box neighborhood for the
%                       summation process.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ..., }.
% Output:
%   - mOutputImage  -   Output Image.
%                       Structure: Image Matrix (Single Channel)
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
% Remarks:
%   1.  References: "..."
%   2.  The running sum matches Intel IPP 'SumWindowRow' / 'SumWindowColumn'.
% TODO:
%   1.  s
%   Release Notes:
%   -   1.2.001     29/04/2016  Royi Avital
%       *   Fixed bug were the border pixels weren't calculated correctly.
%   -   1.2.000     29/04/2016  Royi Avital
%       *   Updated function input.
%   -   1.1.000     04/01/2016  Royi Avital
%       *   Using "Running Sum" instead of Integral Images / Summed Area
%           Table due to numeric issues.
%   -   1.0.000     14/03/2015  Royi Avital
%       *   First release version.
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

boxBlurKernelLength     = (2 * boxRadius) + 1;

mOutputImage    = zeros([(numRows + boxBlurKernelLength - 1), (numCols + boxBlurKernelLength - 1)]);
switch(borderType)
    case(BORDER_TYPE_CONSTANT)
        mInputImage     = padarray(mInputImage, [boxRadius, boxRadius], borderValue, 'both');
    case(BORDER_TYPE_CIRCULAR)
        mInputImage     = padarray(mInputImage, [boxRadius, boxRadius], 'circular', 'both');
    case(BORDER_TYPE_REPLICATE)
        mInputImage     = padarray(mInputImage, [boxRadius, boxRadius], 'replicate', 'both');
    case(BORDER_TYPE_SYMMETRIC)
        mInputImage     = padarray(mInputImage, [boxRadius, boxRadius], 'symmetric', 'both');
end

vRowsIdx = boxRadius + [1:numRows];
vColsIdx = boxRadius + [1:numCols];
removedIdx = boxRadius + 1;

vCurrSum = sum(mInputImage(1:boxBlurKernelLength, :), 1);
mOutputImage((boxRadius + 1), :) = vCurrSum;

for iRow = boxRadius + [2:numRows]
    vRowToAdd       = mInputImage((iRow + boxRadius) ,:);
    vRowToRemove    = mInputImage((iRow - removedIdx) ,:);
    vCurrSum        = vCurrSum + vRowToAdd - vRowToRemove;
    
    mOutputImage(iRow, :) = vCurrSum;
end

mInputImage = mOutputImage;

vCurrSum = sum(mInputImage(:, 1:boxBlurKernelLength), 2);
mOutputImage(:, (boxRadius + 1)) = vCurrSum;

for iCol = boxRadius + [2:numCols]
    vColToAdd       = mInputImage(:, (iCol + boxRadius));
    vColToRemove    = mInputImage(:, (iCol - removedIdx));
    vCurrSum        = vCurrSum + vColToAdd - vColToRemove;
    
    mOutputImage(:, iCol) = vCurrSum;
end

mOutputImage = mOutputImage(vRowsIdx, vColsIdx);

if(normalizeFlag == ON)
    mOutputImage = mOutputImage ./ (boxBlurKernelLength * boxBlurKernelLength);
end


end

