function [ mK ] = CreateImageConvMtx( mH, numRows, numCols, convShape )
% ----------------------------------------------------------------------------------------------- %
%Generates a Convolution Matrix for the 2D Kernel (The Matrix mH) with
%support for different convolution shapes (Full / Same / Valid).
% Input:
%   - mH                -   Input 2D Convolution Kernel.
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
%                           The output convolution matrix. Multiplying in
%                           the column stack form on an image should be
%                           equivalent to applying convolution on the
%                           image.
%                           Structure: Matrix (Sparse).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  MATLAB's 'convmtx2()' - https://www.mathworks.com/help/images/ref/convmtx2.html.
% Remarks:
%   1.  gf
% TODO:
%   1.  
%   Release Notes:
%   -   1.0.000     dd/mm/yyyy  firstName lastName
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

CONVOLUTION_SHAPE_FULL  = 1;
CONVOLUTION_SHAPE_SAME  = 2;
CONVOLUTION_SHAPE_VALID = 3;

switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        % Code for the 'full' case
        convShapeString = 'full';
    case(CONVOLUTION_SHAPE_SAME)
        % Code for the 'same' case
        convShapeString = 'same';
    case(CONVOLUTION_SHAPE_VALID)
        % Code for the 'valid' case
        convShapeString = 'valid';
end

mImpulse = zeros(numRows, numCols);

for ii = numel(mImpulse):-1:1
    
    mImpulse(ii) = 1;  %Create impulse image corresponding to i-th output matrix column
    
    mTmp = sparse(conv2(mImpulse, mH, convShapeString));  %impulse response
    
    cColumn{ii} = mTmp(:);
    
    mImpulse(ii) = 0;
end

mK = cell2mat(cColumn);


end

