function [ mK ] = CreateImageFilterMtx( mH, numRows, numCols, operationMode, boundaryMode )
% ----------------------------------------------------------------------------------------------- %
% [ mK ] = CreateImageFilterMtx( mH, numRows, numCols, operationMode, boundaryMode )
% Generates an Image Filtering Matrix for the 2D Kernel (The Matrix mH)
% with support for different operations modes (Convolution / Correlation)
% and boundary conditions (Zeros, Symmetric, Replicate, Circular). The
% function should match the use of MATLAB's 'imfilter()' with the same
% parameters.
% Input:
%   - mH                -   Input 2D Convolution Kernel.
%                           Assumed to have odd dimensions.
%                           Structure: Matrix.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - numRows           -   Number of Rows.
%                           Number of rows in the output convolution
%                           matrix.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, ...}.
%   - numCols           -   Number of Columns.
%                           Number of columns in the output convolution
%                           matrix.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, ...}.
%   - operationMode     -   Operation Mode.
%                           Sets whether to use Convolution or Correlation
%                           for the operation mode.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2}.
%   - boundaryMode      -   Boundary Condition Mode.
%                           Sets the boundary condition mode for the
%                           filtering. The options are - Zeros, Symmetric,
%                           Replicate and Circular.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, 4}.
% Output:
%   - mK                -   Convolution Matrix.
%                           The output filtering matrix. Multiplying in
%                           the column stack form on an image should be
%                           equivalent to applying 'imfilter()' on the
%                           image.
%                           Structure: Matrix (Sparse).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  MATLAB's 'imfilter()' - https://www.mathworks.com/help/images/ref/imfilter.html.
%   2.  MATLAB's 'convmtx2()' - https://www.mathworks.com/help/images/ref/convmtx2.html.
% Remarks:
%   1.  The height and width of 'mH' are assumed to be odd number. In case
%       either or both are even the user should pad the kernel with zeros
%       (Either a row, column or both). according to the anchor of the
%       kernel the user do the padding pre or post the kernel.
%   2.  Currently it supports only Zero Boundary Condition while MATLAB's
%       'imfilter()' supports any constant as boundary condition.
% TODO:
%   1.  Refactor the code to share the common operations of different
%       boundary modes.
%   2.  Add support for any constant as boundary condition and not only 0.
% Release Notes:
%   -   1.0.001     30/12/2019  Royi Avital
%       *   Fixed some bugs related to using 'numCols' instead of 'numRows'
%           in the calculation of 'pixelShift' for the cases 'jj + ll >
%           numCols' and 'jj + ll < 1'.
%   -   1.0.000     16/01/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

OPERATION_MODE_CONVOLUTION = 1;
OPERATION_MODE_CORRELATION = 2;

BOUNDARY_MODE_ZEROS         = 1;
BOUNDARY_MODE_SYMMETRIC     = 2;
BOUNDARY_MODE_REPLICATE     = 3;
BOUNDARY_MODE_CIRCULAR      = 4;

switch(operationMode)
    case(OPERATION_MODE_CONVOLUTION)
        mH = mH(end:-1:1, end:-1:1);
    case(OPERATION_MODE_CORRELATION)
        % mH = mH; %<! Default Code is correlation
end

switch(boundaryMode)
    case(BOUNDARY_MODE_ZEROS)
        mK = CreateConvMtxZeros(mH, numRows, numCols);
    case(BOUNDARY_MODE_SYMMETRIC)
        mK = CreateConvMtxSymmetric(mH, numRows, numCols);
    case(BOUNDARY_MODE_REPLICATE)
        mK = CreateConvMtxReplicate(mH, numRows, numCols);
    case(BOUNDARY_MODE_CIRCULAR)
        mK = CreateConvMtxCircular(mH, numRows, numCols);
end


end


function [ mK ] = CreateConvMtxZeros( mH, numRows, numCols )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

numElementsImage    = numRows * numCols;
numRowsKernel       = size(mH, 1);
numColsKernel       = size(mH, 2);
numElementsKernel   = numRowsKernel * numColsKernel;

vRows = reshape(repmat(1:numElementsImage, numElementsKernel, 1), numElementsImage * numElementsKernel, 1);
vCols = zeros(numElementsImage * numElementsKernel, 1);
vVals = zeros(numElementsImage * numElementsKernel, 1);

kernelRadiusV = floor(numRowsKernel / 2);
kernelRadiusH = floor(numColsKernel / 2);

pxIdx       = 0;
elmntIdx    = 0;

for jj = 1:numCols
    for ii = 1:numRows
        pxIdx = pxIdx + 1;
        for ll = -kernelRadiusH:kernelRadiusH
            for kk = -kernelRadiusV:kernelRadiusV
                elmntIdx = elmntIdx + 1;
                
                % Pixel Index Shift such that pxIdx + pxShift is the linear
                % index of the pixel in the image
                pxShift = (ll * numRows) + kk;
                
                if((ii + kk <= numRows) && (ii + kk >= 1) && (jj + ll <= numCols) && (jj + ll >= 1))
                    vCols(elmntIdx) = pxIdx + pxShift;
                    vVals(elmntIdx) = mH(kk + kernelRadiusV + 1, ll + kernelRadiusH + 1);
                else
                    vCols(elmntIdx) = pxIdx;
                    vVals(elmntIdx) = 0; % See the accumulation property of 'sparse()'.
                end
            end
        end
    end
end

mK = sparse(vRows, vCols, vVals, numElementsImage, numElementsImage);


end


function [ mK ] = CreateConvMtxSymmetric( mH, numRows, numCols )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

numElementsImage    = numRows * numCols;
numRowsKernel       = size(mH, 1);
numColsKernel       = size(mH, 2);
numElementsKernel   = numRowsKernel * numColsKernel;

vRows = reshape(repmat(1:numElementsImage, numElementsKernel, 1), numElementsImage * numElementsKernel, 1);
vCols = zeros(numElementsImage * numElementsKernel, 1);
vVals = zeros(numElementsImage * numElementsKernel, 1);

kernelRadiusV = floor(numRowsKernel / 2);
kernelRadiusH = floor(numColsKernel / 2);

pxIdx       = 0;
elmntIdx    = 0;

for jj = 1:numCols
    for ii = 1:numRows
        pxIdx = pxIdx + 1;
        for ll = -kernelRadiusH:kernelRadiusH
            for kk = -kernelRadiusV:kernelRadiusV
                elmntIdx = elmntIdx + 1;
                
                % Pixel Index Shift such that pxIdx + pxShift is the linear
                % index of the pixel in the image
                pxShift = (ll * numRows) + kk;
                
                if(ii + kk > numRows)
                    pxShift = pxShift - (2 * (ii + kk - numRows) - 1);
                end
                
                if(ii + kk < 1)
                    pxShift = pxShift + (2 * (1 -(ii + kk)) - 1);
                end
                
                if(jj + ll > numCols)
                    pxShift = pxShift - ((2 * (jj + ll - numCols) - 1) * numRows);
                end
                
                if(jj + ll < 1)
                    pxShift = pxShift + ((2 * (1 - (jj + ll)) - 1) * numRows);
                end
                
                vCols(elmntIdx) = pxIdx + pxShift;
                vVals(elmntIdx) = mH(kk + kernelRadiusV + 1, ll + kernelRadiusH + 1);
                
            end
        end
    end
end

mK = sparse(vRows, vCols, vVals, numElementsImage, numElementsImage);


end


function [ mK ] = CreateConvMtxReplicate( mH, numRows, numCols )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

numElementsImage    = numRows * numCols;
numRowsKernel       = size(mH, 1);
numColsKernel       = size(mH, 2);
numElementsKernel   = numRowsKernel * numColsKernel;

vRows = reshape(repmat(1:numElementsImage, numElementsKernel, 1), numElementsImage * numElementsKernel, 1);
vCols = zeros(numElementsImage * numElementsKernel, 1);
vVals = zeros(numElementsImage * numElementsKernel, 1);

kernelRadiusV = floor(numRowsKernel / 2);
kernelRadiusH = floor(numColsKernel / 2);

pxIdx       = 0;
elmntIdx    = 0;

for jj = 1:numCols
    for ii = 1:numRows
        pxIdx = pxIdx + 1;
        for ll = -kernelRadiusH:kernelRadiusH
            for kk = -kernelRadiusV:kernelRadiusV
                elmntIdx = elmntIdx + 1;
                
                % Pixel Index Shift such that pxIdx + pxShift is the linear
                % index of the pixel in the image
                pxShift = (ll * numRows) + kk;
                
                if(ii + kk > numRows)
                    pxShift = pxShift - (ii + kk - numRows);
                end
                
                if(ii + kk < 1)
                    pxShift = pxShift + (1 - (ii + kk));
                end
                
                if(jj + ll > numCols)
                    pxShift = pxShift - ((jj + ll - numCols) * numRows);
                end
                
                if(jj + ll < 1)
                    pxShift = pxShift + ((1 - (jj + ll)) * numRows);
                end
                
                vCols(elmntIdx) = pxIdx + pxShift;
                vVals(elmntIdx) = mH(kk + kernelRadiusV + 1, ll + kernelRadiusH + 1);
                
            end
        end
    end
end

mK = sparse(vRows, vCols, vVals, numElementsImage, numElementsImage);


end


function [ mK ] = CreateConvMtxCircular( mH, numRows, numCols )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

numElementsImage    = numRows * numCols;
numRowsKernel       = size(mH, 1);
numColsKernel       = size(mH, 2);
numElementsKernel   = numRowsKernel * numColsKernel;

vRows = reshape(repmat(1:numElementsImage, numElementsKernel, 1), numElementsImage * numElementsKernel, 1);
vCols = zeros(numElementsImage * numElementsKernel, 1);
vVals = zeros(numElementsImage * numElementsKernel, 1);

kernelRadiusV = floor(numRowsKernel / 2);
kernelRadiusH = floor(numColsKernel / 2);

pxIdx       = 0;
elmntIdx    = 0;

for jj = 1:numCols
    for ii = 1:numRows
        pxIdx = pxIdx + 1;
        for ll = -kernelRadiusH:kernelRadiusH
            for kk = -kernelRadiusV:kernelRadiusV
                elmntIdx = elmntIdx + 1;
                
                % Pixel Index Shift such that pxIdx + pxShift is the linear
                % index of the pixel in the image
                pxShift = (ll * numRows) + kk;
                
                if(ii + kk > numRows)
                    pxShift = pxShift - numRows;
                end
                
                if(ii + kk < 1)
                    pxShift = pxShift + numRows;
                end
                
                if(jj + ll > numCols)
                    pxShift = pxShift - (numCols * numRows);
                end
                
                if(jj + ll < 1)
                    pxShift = pxShift + (numCols * numRows);
                end
                
                vCols(elmntIdx) = pxIdx + pxShift;
                vVals(elmntIdx) = mH(kk + kernelRadiusV + 1, ll + kernelRadiusH + 1);
                
            end
        end
    end
end

mK = sparse(vRows, vCols, vVals, numElementsImage, numElementsImage);


end

