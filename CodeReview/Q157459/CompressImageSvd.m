function [ mO ] = CompressImageSvd( mI, energyThr, blockRadius )
% ----------------------------------------------------------------------------------------------- %
% [ mO ] = CompressImageSvd( mI, energyThr, blockRadius )
%   Compresses the image using SVD.
% Input:
%   - mI            -   Input Image.
%                       Structure: Image Matrix (Signle Channel or RGB).
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
%   - energyThr     -   Energy Threshold.
%                       Sets the threshold for Singular Value kept energy.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
%   - blockRadius   -   Block Radius.
%                       Sets the block radius for the compression.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ...}.
% Output:
%   - mO            -   Output Image.
%                       Structure: Image Matrix (Signle Channel or RGB).
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
% References
%   1.  SVD Wikipedia - https://en.wikipedia.org/wiki/Singular_value_decomposition.
% Remarks:
%   1.  a
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     01/09/2017  Royi
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numRows = size(mI, 1);
numCols = size(mI, 2);
numChan = size(mI, 3); %<! Num Channels

vImageDim = [numRows, numCols];

blockLength = (2 * blockRadius) + 1;
vBlockDim   = [blockLength, blockLength];

mO = zeros([numRows, numCols, numChan]);

for ii = 1:numChan
    
    mII     = mI(:, :, ii);
    dcLevel = mean(mII(:)); %<! Extracting DC Level
    mII     = mII - dcLevel;
    
    % Decomposing the image into blocks. Each block becomes a vector in the
    % Columns Images.
    mColImage   = im2col(mII, vBlockDim, 'distinct');
    
    % The SVD Step
    [mU, mS, mV] = svd(mColImage);
    
    vSingularValues = diag(mS);
    
    vSingularValueEnergy = cumsum(vSingularValues) / sum(vSingularValues);
    lastIdx = find(vSingularValueEnergy >= energyThr, 1, 'first');
    
    vSingularValues(lastIdx + 1:end) = 0;
    % mS isn't necessarily square matrix. Hence only work on its main
    % diagonal.
    mS(1:length(vSingularValues), 1:length(vSingularValues)) = diag(vSingularValues);
    
    % Reconstruction of the image using "Less Energy".
    mColImage = mU * mS * mV.';
    
    % Restorig the original structure and the DC Level
    mO(:, :, ii) = col2im(mColImage, vBlockDim, vImageDim, 'distinct') + dcLevel;
    
end


end

