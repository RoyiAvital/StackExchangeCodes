function [ mColumnVectorImage ] = ImageToColumnsSliding( mInputImage, vBlockSize )
% ----------------------------------------------------------------------------------------------- %
% [ mColumnImage ] = ImageToColumns( mInputImage, blockRadius )
%   Creates an column image from the sliding neighborhood in mInpuImage
% Input:
%   - mInputImage           -   Input image.
%                               Structure: Image Matrix (Single Channel)
%                               Type: 'Single' / 'Double'.
%                               Range: [0, 1].
%   - vBlockSize            -   Block Size.
%                               Structure: 2D Vector.
%                               Type: 'Single' / 'Double'.
%                               Range: [0, 1].
% Output:
%   - mColumnVectorImage    -   Column Vector Image.
%                               Structure: Image Matrix (Single Channel)
%                               Type: 'Single' / 'Double'.
%                               Range: [0, 1].
% Remarks:
%   1.  Prefixes:
%       -   'm' - Matrix.
%       -   'v' - Vector.
%   2.  Converts each sliding `vBlockSize(1)` by `vBlockSize(2) block of
%       `mInputImage` into a column of `mColumnVectorImage` with no zero
%       padding.
%   3.  Shouldn't be used for images larger than 400x400 and blocks of
%       51x51.
% TODO:
%   1.  I
%   Release Notes:
%   -   1.0.000     20/03/2015  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %


[numRows, numCol] = size(mInputImage);
blockNumRows = vBlockSize(1);
blockNumCols = vBlockSize(2);

% Create Hankel-like indexing sub matrix.
nc = numRows - blockNumRows + 1;
nn = numCol - blockNumCols + 1;

vColumnIdx = [(0:(blockNumRows - 1))]';
vRowIdx = [1:nc];

t = vColumnIdx(:, ones(nc, 1)) + vRowIdx(ones(blockNumRows, 1), :);    % Hankel Subscripts
tt = zeros(blockNumRows * blockNumCols, nc);
rows = 1:blockNumRows;
for ii = 0:(blockNumCols - 1)
    tt(((ii * blockNumRows) + rows), :) = t + (numRows * ii);
end
mColumnVectorImageIdx = zeros((blockNumRows * blockNumCols), (nc * nn));
cols = 1:nc;
for jj = 0:(nn - 1)
    mColumnVectorImageIdx(:, ((jj * nc) + cols)) = tt + (numRows * jj);
end

mColumnVectorImage = mInputImage(mColumnVectorImageIdx);


end

