
clear();

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;

numSamples  = 9;
convType    = CONVOLUTION_SHAPE_VALID;
paramLambda = 0;

vH = [1; 2; 3];
vG = [1; -1];

vX = rand(numSamples, 1);

mH = full(CreateConvMtx1D(vH, numSamples, convType));
mG = full(CreateConvMtx1D(vG, numSamples, convType));

vY  = ((mH.' * mH) + (paramLambda * (mG.' * mG))) * vX;
vYY = ConvNormalEquations(vX, vH, vG, convType, paramLambda);

max(abs(vYY - vY))