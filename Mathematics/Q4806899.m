% https://math.stackexchange.com/questions/4806899
clear();

numRounds   = 10;
paramGamma  = 1.5;
valP0       = 0.95;
valC        = 1000;


mM = zeros(numRounds + 1, numRounds + 1);
mM(1, :) = valC;
mM(:, 1) = valC;

ii      = 1;
valPi   = valP0;
for kk = 1:numRounds
    ii = ii + 1;
    valPi = valPi * valPi;
    for jj = 2:ii
        mM(ii, jj) = max(mM(ii - 1, jj), paramGamma * valPi * mM(ii - 1, jj - 1));
    end
end