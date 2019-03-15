function [ mHC ] = CircularExtension2D( mH, numRows, numCols )

kernelRadiusV = floor(size(mH, 1) / 2); %<! Vertical Radius
kernelRadiusH = floor(size(mH, 2) / 2); %<! Horizontal Radius

mHC = mH;
mHC(numRows, numCols) = 0;
mHC = circshift(mHC, [-kernelRadiusV, -kernelRadiusH]);


end