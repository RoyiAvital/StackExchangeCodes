function [ vH ] = EstimateFilterCoeff( vH, vX, vY, convShape, numIterations, stopThr )
% ----------------------------------------------------------------------------------------------- %
% [ mK ] = CreateConvMtx1D( vK, numElements, convShape )
% Generates a Convolution Matrix for 1D Kernel (The Vector vK) with
% support for different convolution shapes (Full / Same / Valid). The
% matrix is build such that for a signal 'vS' with 'numElements = size(vS
% ,1)' the following are equiavlent: 'mK * vS' and conv(vS, vK,
% convShapeString);
% Input:
%   - vK                -   Input 1D Convolution Kernel.
%                           Structure: Vector.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - numElements       -   Number of Elements.
%                           Number of elements of the vector to be
%                           convolved with the matrix. Basically set the
%                           number of columns of the Convolution Matrix.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, ...}.
%   - convShape         -   Convolution Shape.
%                           The shape of the convolution which the output
%                           convolution matrix should represent. The
%                           options should match MATLAB's conv2() function
%                           - Full / Same / Valid.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3}.
% Output:
%   - mK                -   Convolution Matrix.
%                           The output convolution matrix. The product of
%                           'mK' and a vector 'vS' ('mK * vS') is the
%                           convolution between 'vK' and 'vS' with the
%                           corresponding convolution shape.
%                           Structure: Matrix (Sparse).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  MATLAB's 'convmtx()' - https://www.mathworks.com/help/signal/ref/convmtx.html.
% Remarks:
%   1.  The output matrix is sparse data type in order to make the
%       multiplication by vectors to more efficient.
%   2.  In caes the same convolution is applied on many vectors, stacking
%       them into a matrix (Each signal as a vector) and applying
%       convolution on each column by matrix multiplication might be more
%       efficient than applying classic convolution per column.
% TODO:
%   1.  
%   Release Notes:
%   -   1.0.000     20/01/2019  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;

switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        hGradFun = @(vH) CalcConvGradSame(vH, vX, vY);
    case(CONVOLUTION_SHAPE_SAME)
        hGradFun = @(vH) CalcConvGradSame(vH, vX, vY);
        convShapeString = 'same';
    case(CONVOLUTION_SHAPE_VALID)
        hGradFun = @(vH) conv2(vX(end:-1:1), (conv2(vX, vH, 'valid') - vY), 'valid');
        convShapeString = 'valid';
end

for ii = 1:numIterations
    vHPrev = vH;
    
    vG = hGradFun(vH);
    hObjFun = @(stepSize) sum((conv(vX, vH - (stepSize * vG), convShapeString) - vY) .^ 2);
    stepSize = fminbnd(hObjFun, 0, 2);
    vH = vH - (stepSize * vG);
    vH = ProjectSimplex(vH, 1, 1e-6);
    
    if(max(abs(vH - vHPrev)) < stopThr)
        break;
    end
end


end


function [ vD ] = CalcConvGradSame( vH, vX, vY )

numCoefficients = size(vH, 1);
numSamples = size(vX, 1);

vDD = conv2(vX(end:-1:1), (conv2(vX, vH, 'same') - vY), 'full');
% vD = vD((numCoefficients + 1):(numCoefficients + numCoefficients));

vD = zeros(numCoefficients, 1);
if(numCoefficients < (2 * numSamples - 1))
    firstIdx = floor((size(vDD, 1) - numCoefficients) / 2) + 1;
    lastIdx = firstIdx + size(vD, 1) - 1;
    vD = vDD(firstIdx:lastIdx);
else
    firstIdx = ceil((numCoefficients - size(vDD, 1)) / 2) + 1;
    lastIdx = firstIdx + size(vDD, 1) - 1;
    vD(firstIdx:lastIdx) = vDD;
end


end

