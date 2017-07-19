function [ mO ] = ApplyLocalLinearKernel( mI, filterRadius, regFctr )
% ----------------------------------------------------------------------------------------------- %
%[ mOutputImage ] = ApplyLocalLinearFilter( mInputImage, filterRadius, regFctr )
% Applying Linear Edge Preserving Smoothing Filter based on Local Linear (Affine) model.
% Input:
%   - mI            -   Input Image.
%                       Structure: Image Matrix (Single Channel).
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
%   - filterRadius  -   Filter Radius.
%                       The filter radius.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ...}.
%   - regFctr       -   Regularization Factor.
%                       Regularize the local covariance (Variance).
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
% Output:
%   - mO            -   Output Image.
%                       Structure: Image Matrix (Single Channel).
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
% References:
%   1.  "Guided Image Filtering".
% Remarks:
%   1.  This is basically estimating the Linear Function (Affine)
%       parameters for the Local Window. Namely, the ouput is a Linear
%       combination of the input Window and a DC Factor. The final step is
%       aggregation (Uniform) off all estimations of the parameters.
%   2.  Prefixes:
%       -   'v' - Vector.
%       -   'm' - Matrix.
%       -   't' - Tensor (Multi Dimension Matrix)
%       -   's' - Struct.
%       -   'c' - Cell Array.
%   3.  The calculation of the Local Variance might be negative due to
%       numerical diffuculties. If artifacts appear, this might be the
%       casue. Usually using matrices of type 'double' solves it.
%   4.  This implemnetation is `ApplyGuidedFilter` where the Guiding Image
%       is the Input Image.
%   5.  Speed otimizzation can be achived by wiser use of 'mNumEffPixels'.
%       Instead of dividing by it calculate its reciprocal once. Moreover,
%       it can be used only once in the aggregation process.
% TODO:
%   1.  Create Multi Variable Linear Model.
%   2.  Some speed potimization could be made (Taking advantage of 'mNumEffPixels').
% Release Notes:
%   -   1.0.000     05/01/2016  Royi Avital
%       *   First release version
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF = 0;
ON  = 1;

BORDER_TYPE_CONSTANT    = 1;
BORDER_TYPE_CIRCULAR    = 2;
BORDER_TYPE_REPLICATE   = 3;
BORDER_TYPE_SYMMETRIC   = 4;

numRows = size(mI, 1);
numCols = size(mI, 2);

borderType      = BORDER_TYPE_CONSTANT;
borderValue     = 0;
normalizeFlag   = OFF;

mNumEffPixels = ApplyBoxFilter(ones([numRows, numCols]), filterRadius, borderType, borderValue, normalizeFlag);

mLocalMean          = ApplyBoxFilter(mI, filterRadius, borderType, borderValue, normalizeFlag) ./ mNumEffPixels;
mLocalMeanSquare    = ApplyBoxFilter((mI .* mI), filterRadius, borderType, borderValue, normalizeFlag) ./ mNumEffPixels;
mLocalCovariance    = mLocalMeanSquare - (mLocalMean .* mLocalMean);

mO = zeros([numRows, numCols]);

for jj = 1:numCols
    for ii = 1:numRows
        % The coordinate i, j are set.
        
        sumWeights = 0;
        
        % Running on the Neighborhood of the pixel
        for ll = -filterRadius:filterRadius
            for kk = -filterRadius:filterRadius
                
                jRowIdx = ii + kk;
                jColIdx = jj + ll;
                
                if((jColIdx >= 1) && (jColIdx <= numCols) && (jRowIdx >= 1) && (jRowIdx <= numRows))
                    % Valid Pixel
                    numPixels       = 0;
                    jPixelWeight    = 0;
                    
                    % Running on Wk Window
                    for nn = -filterRadius:filterRadius
                        for mm = -filterRadius:filterRadius
                            
                            % K Index
                            kRowIdx = jRowIdx + mm;
                            kColIdx = jColIdx + nn;
                            
                            if((kColIdx >= 1) && (kColIdx <= numCols) && (kRowIdx >= 1) && (kRowIdx <= numRows) && (abs(kColIdx - jj) <= filterRadius) && (abs(kRowIdx - ii) <= filterRadius))
                                
                                numPixels = numPixels + 1;
                                
                                % Weight
                                jPixelWeight = jPixelWeight + 1 + (((mI(ii, jj) - mLocalMean(kRowIdx, kColIdx)) * (mI(jRowIdx, jColIdx) - mLocalMean(kRowIdx, kColIdx))) / (mLocalCovariance(kRowIdx, kColIdx) + regFctr));
                            end
                        end
                    end
                    
                    % jPixelWeight = jPixelWeight / (numPixels * numPixels);
                    mO(ii, jj) = mO(ii, jj) + (jPixelWeight * mI(jRowIdx, jColIdx));
                    
                    sumWeights = sumWeights + jPixelWeight;
                    
                end
            end
        end
        
        mO(ii, jj) = mO(ii, jj) / sumWeights;
        
    end
end



end

