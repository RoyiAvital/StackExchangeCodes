function [ mO ] = PadArrayReplicate( mI, padRadius )
%PADARRAYREPLICATE Summary of this function goes here
%   Detailed explanation goes here

numRows     = size(mI, 1);
numCols     = size(mI, 2);
numChannels = size(mI, 3);

if(numChannels == 1)
    vImageSize = [numRows; numCols];
    vPadRadius = [padRadius; padRadius];
elseif(numChannels == 3)
    vImageSize = [numRows; numCols; numChannels];
    vPadRadius = [padRadius; padRadius; 0];
end

cPadIdx = cell(1, length(vImageSize));

for ii = 1:length(vImageSize)
    dimSize     = vImageSize(ii);
    padRadius   = vPadRadius(ii); 
    vE          = uint32(ones(1, padRadius));
    cPadIdx{ii} = [vE, 1:dimSize, (dimSize * vE)];
end

mO = mI(cPadIdx{:});


end

