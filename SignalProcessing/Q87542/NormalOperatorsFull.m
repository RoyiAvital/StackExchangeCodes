
clear();

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;

% Type -> Adjoint
% CONVOLUTION_SHAPE_FULL -> CONVOLUTION_SHAPE_SAME
% CONVOLUTION_SHAPE_SAME -> 
% CONVOLUTION_SHAPE_VALID -> 

numSamples  = 9;
convType    = CONVOLUTION_SHAPE_FULL;
convTypeAdj = CONVOLUTION_SHAPE_SAME;

vH = [1; 2; 3; 4];
mH = full(CreateConvMtx1D(vH, numSamples, convType));

vHH = conv(vH, flip(vH));
mHH = mH.' * mH;

mT = full(CreateConvMtx1D(vHH, numSamples, convTypeAdj));

max(abs(mT - mHH), [], 'all')