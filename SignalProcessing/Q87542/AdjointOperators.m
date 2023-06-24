
clear();

CONVOLUTION_SHAPE_FULL  = 1;
CONVOLUTION_SHAPE_SAME  = 2;
CONVOLUTION_SHAPE_VALID = 3;

% Type -> Adjoint
% CONVOLUTION_SHAPE_FULL -> CONVOLUTION_SHAPE_SAME
% CONVOLUTION_SHAPE_SAME -> 
% CONVOLUTION_SHAPE_VALID -> 

numSamples  = 9;
convType    = CONVOLUTION_SHAPE_VALID;

vH = [1; 2; 3; 4];
vX = rand(numSamples, 1);

switch(convType)
    case(CONVOLUTION_SHAPE_FULL)
        convTypeAdj     = CONVOLUTION_SHAPE_VALID;
        numSamplesAdj   = numSamples + length(vH) - 1;
        convTypeStr     = 'full';
        vHCorr          = flip(vH);
    case(CONVOLUTION_SHAPE_SAME)
        convTypeAdj     = CONVOLUTION_SHAPE_SAME;
        numSamplesAdj   = numSamples;
        convTypeStr     = 'same';
        vHCorr          = flip([vH; 0]);
    case(CONVOLUTION_SHAPE_VALID)
        convTypeAdj     = CONVOLUTION_SHAPE_FULL;
        numSamplesAdj   = numSamples - length(vH) + 1;
        convTypeStr     = 'valid';
        vHCorr          = flip(vH);
end


mH  = full(CreateConvMtx1D(vH, numSamples, convType));
max(abs(conv(vX, vH, convTypeStr) - mH * vX))

mHt = full(CreateConvMtx1D(vHCorr, numSamplesAdj, convTypeAdj));
max(abs(mHt - mH.'), [], 'all')

