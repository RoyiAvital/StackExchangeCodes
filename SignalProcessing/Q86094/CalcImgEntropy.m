function [ imgEntropy ] = CalcImgEntropy( mI, entropyMode )
% ----------------------------------------------------------------------------------------------- %
% [ imgEntropy ] = CalcImgEntropy( mI, entropyMode )
% Calculates the entropy of an image with up to 8 channels. Supports per
% channel clauclation or represenation of the pixel as a single number.
% Input:
%   - mI                -   Input Image.
%                           The image to have its entropy calculated.
%                           Structure: Matrix (numRowx x numCols).
%                           Type: 'unit8'.
%                           Range: [0, 255].
%   - entropyMode       -   Entropy Mode.
%                           Sets the entropy calculation mode: Per channel
%                           or as an integrated pixel value.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2}.
% Output:
%   - imgEntropy        -   The Image Entropy.
%                           The calculated entropy value.
%                           Structure: Scalar (1 x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  A
% Remarks:
%   1.  The "Vector" calculation basically concatentate each pixel value
%       in its bits representation. For instance, for RGB image of input
%       `uint8` we basically creates a value of 24 Bits by shifting the G
%       channel by 8 bits and the B channel by 16 bits.
%   2.  Remove the `areguments` blocks for compatibility with MATLAB
%       versions prior to R2022b.
% TODO:
%   1.  Add support for `uint16` input with up to 3 channels.
% Release Notes:
%   -   1.0.000     10/10/2022  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments (Input)
    mI (:, :, :) {mustBeA(mI, {'uint8'})}
    entropyMode (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger, mustBeMember(entropyMode, [1, 2])} = 1
end

arguments (Output)
    imgEntropy (1, 1) {mustBeNumeric, mustBeReal, mustBeNonnegative}
end

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

ENTROPY_MODE_CHANNEL    = 1; %<! Averages per channel calculation
ENTROPY_VECTOR          = 2; %<! Treats the RGB data as a single vector per pixel

numRows     = size(mI, 1);
numCols     = size(mI, 2);
numChannels = size(mI, 3);

if(numChannels > 8)
    % 8 channels of `uint8` can be packed into `uint64`
    error('Input Image Must Have # Channels < 8');
end

if(numChannels == 1)
    imgEntropy = CalcEntropy(mI);
else
    if(entropyMode == ENTROPY_MODE_CHANNEL)
        imgEntropy = 0;
        for ii = 1:numChannels
            imgEntropy = imgEntropy + CalcEntropy(mI(:, :, ii));
        end
        imgEntropy = imgEntropy / numChannels;
    elseif(entropyMode == ENTROPY_VECTOR)
        if(numChannels > 4)
            mI = uint64(mI);
        else
            mI = uint32(mI);
        end
        mD = mI(:, :, 1);
        for ii = 2:numChannels
            mD = mD + bitshift(mI(:, :, ii), 8 * (ii - 1));
        end
        imgEntropy = CalcEntropy(mD(:));
    end
end


end


function [ valEntropy ] = CalcEntropy( vI ) 

arguments
    vI (:, :) {mustBeA(vI, {'uint8', 'uint16', 'uint32', 'uint64'})}
end

% See https://www.mathworks.com/matlabcentral/answers/37196
% See https://www.mathworks.com/matlabcentral/answers/96504

vU = unique(vI(:));
vP = histc(vI(:), vU);

vP = vP / sum(vP); %<! Make it probability

valEntropy = -sum(vP .* log2(vP));


end

