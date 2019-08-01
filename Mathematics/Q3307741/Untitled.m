clc();

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;
DIFF_MODE_COMPLEX   = 4;

numRows = 3;
numCols = 2;

numRowsX = numCols;
numColsX = 2 * numRows;

mA = randn(numRows, numCols);
vX = randn(numCols, 1);

hMixedNorm = @(vX) norm(mA * vX);
hMixedNorm = @(vX) sqrt(sum((mA * vX) .^ 2)); %<! COmplex Mode Friendly

diffMode = DIFF_MODE_COMPLEX;
epsVal = 1e-6;


vG = CalcFunGrad(vX, hMixedNorm, diffMode, epsVal);
mA.' * mA * vX ./ (norm(mA * vX));

mX = rand(numCols, 2 * numRows);
vX = mX(:);
hMixedNorm = @(vX) MixedNormL21(vX, mA, numRowsX, numColsX);

vG = CalcFunGrad(vX, hMixedNorm, diffMode, epsVal)

reshape(mA.' * mA * mX ./ sqrt(sum( (mA * mX) .^ 2 )), [], 1)



function [ normVal ] = MixedNormL21( vX, mA, numRows, numCols )

normVal = sum(sqrt(sum( (mA * reshape(vX, numRows, numCols)) .^ 2, 1)));

end

function [ normVal ] = CreateDMatrix( vX, mA, numRows, numCols )

normVal = sum(sqrt(sum( (mA * reshape(vX, numRows, numCols)) .^ 2, 1)));

end

