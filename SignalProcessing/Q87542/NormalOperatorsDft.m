
clear();

numSamples = 5;
numCoeff   = 3;

numSamplesExt = numSamples;

mE = eye(numSamplesExt);
mE = mE(numCoeff:numSamplesExt, :);

mP = eye(numSamples);
% mP(numSamplesExt, numSamples) = 0;

vH = [1; 2; 3];
vX = [0; 1; 2; 3; 4];

mH = CreateConvMtx1DCyclic(vH, numSamplesExt);

% mH * mP * vX - cconv(vH, mP * vX, numSamplesExt)

mP.' * mH.' * mE.' * mE * mH * mP

mP.' * mH.' * mH * (mE.' * mE) * mP