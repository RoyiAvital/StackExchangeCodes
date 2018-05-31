function [ mO ] = PadArrayCircular( mI, padRadius )
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
    vE          = uint32(1:dimSize);
    cPadIdx{ii} = vE(mod(-padRadius:dimSize + padRadius - 1, dimSize) + 1);
end

mO = mI(cPadIdx{:});


end

