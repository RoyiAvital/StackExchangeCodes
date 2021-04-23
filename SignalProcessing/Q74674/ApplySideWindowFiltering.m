function [ mO ] = ApplySideWindowFiltering( mI, boxRadius, numIterations )
% ----------------------------------------------------------------------------------------------- %
% [ mO ] = ApplySideWindowFiltering( mI, boxRadius, numIterations )
%   Applying Image Edge Preserving Filter using the Side Window Box
%   Filtering algorithm.
% Input:
%   - mI            -   Input Image.
%                       Structure: Image Matrix (Single Channel)
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
%   - boxRadius     -   Box Radius.
%                       Sets the radius of the box filter.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, 3, ...}.
%   - numIterations -   Number of Iterations.
%                       Sets the number of iterations to apply the filter.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, 3, ...}.
% Output:
%   - mO            -   Output Image.
%                       Structure: Image Matrix (Single Channel)
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
% References
%   1.  Side Window Filtering (https://arxiv.org/abs/1905.07177).
% Remarks:
%   1.  Can be much faster if written In Place.
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     24/04/2021  Royi Avital     RoyiAvital@yahoo.com
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

dataType    = class(mI);
boxLength   = (2 * boxRadius) + 1;

% Atoms of the Filter Bank (Since any of the filters is separable)
vK  = ones(boxLength, 1, dataType) / boxLength;
vKU = ones(boxLength, 1, dataType) / (boxRadius + 1); %<! Up
vKD = ones(boxLength, 1, dataType) / (boxRadius + 1); %<! Down

vKU((boxRadius + 2):boxLength)  = 0;
vKD(1:boxRadius)                = 0;

mO = padarray(mI, [boxRadius, boxRadius], 'both', 'replicate');
mF = zeros(size(mO, 1), size(mO, 2), 8, dataType);
mM = zeros(size(mO), dataType); %<! Minimum Index

for kk = 1:numIterations
    % Written for clarity, not performance
    mF(:, :, 1) = conv2(vK, vKU, mO, 'same'); %<! Left Box
    mF(:, :, 2) = conv2(vK, vKD, mO, 'same'); %<! Right Vox
    mF(:, :, 3) = conv2(vKU, vK, mO, 'same'); %<! Up Box
    mF(:, :, 4) = conv2(vKD, vK, mO, 'same'); %<! Down Box
    mF(:, :, 5) = conv2(vKU, vKU, mO, 'same'); %<! NW Box
    mF(:, :, 6) = conv2(vKU, vKD, mO, 'same'); %<! NE Box
    mF(:, :, 7) = conv2(vKD, vKU, mO, 'same'); %<! SW Box
    mF(:, :, 8) = conv2(vKD, vKD, mO, 'same'); %<! SE Box
    
    [~, mM(:, :)]   = min(abs(mF - mO), [], 3); %<! Index which minimizes
    for jj = 1:size(mO, 2)
        for ii = 1:size(mO, 1)
            mO(ii, jj) = mF(ii, jj, mM(ii, jj));
        end
    end
    
end

% Removing boundary
mO = mO((boxRadius + 1):(end - boxRadius), (boxRadius + 1):(end - boxRadius));
mO = min(max(mO, 0), 1);


end

