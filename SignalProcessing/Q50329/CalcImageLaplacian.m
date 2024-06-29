function [ mO ] = CalcImageLaplacian( mI )
% ----------------------------------------------------------------------------------------------- %
%[ mO ] = CalcImageLaplacian( mI )
% Calculates the Laplacian of the Image.
% Input:
%   - mI                -   Input Image.
%                           Structure: Image Matrix (Single Channel).
%                           Type: 'Single' / 'Double'.
%                           Range: [0, 1].
% Output:
%   - mO                -   Output Image.
%                           The Laplacian of the Image.
%                           Structure: Image Matrix (Single Channel).
%                           Type: 'Single' / 'Double'.
%                           Range: [0, 1].
% References
%   1.  Wikipedia Finite Differences Coefficients - https://en.wikipedia.org/wiki/Finite_difference_coefficient.
% Remarks:
%   1.  The Laplacian is given by d^2I/dx^2 + d^2I/dy^2. The second
%       derivative in coefficients should be [1, -2, 1]. Yet this function
%       implements [-1, 2, -1]. This is due the fast usually we are after
%       convolving {D}^{T} D x where D is Derivative Operator with the
%       convolution kernel [1, -1]. So the function implements Dx.' * Dx +
%       Dy.' * Dy. Where D.' is correlation instead of convolution.
% TODO:
%   1.  See if it can be done in one convolution by padding the image and
%       using [-1, 2, 1] directly.
% Release Notes:
%   -   1.0.000     07/07/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

% If mD is the matrix form of convolution by vK then:
%   *   the operation 'reshape(mD * mI(:), numRows - 1, numCols)' is
%       equivalent of 'conv2(mI, vK, 'valid')'.
%   *   The operation of mD.' is correlation. Namely convolving by the
%       flipped kernel (Though dimensions doesn't fit).
%   *   The operation 'reshape(mD.' * mD * mI(:), numRows - 1, numCols)' is
%       equivalent of 'conv2(conv2(mI, vK, 'valid'), vK(end:-1:1), 'full')'
%       namely convolution and then correlation with different Convolution
%       Shape.
%   *   If one uses Full Convolution (Input and Output size are the same)
%       all works with no dimension adjustments.

vK = [1, -1]; %<! Horizontal 1st Derivative
mO = conv2(conv2(mI, vK, 'valid'), vK(end:-1:1), 'full') + conv2(conv2(mI, vK.', 'valid'), flip(vK.', 1), 'full');

% vK = [-1, 2, 1]; %<! Horizontal 2nd Derivative
% mO = conv2(mI, vK) + conv2(mI, vK.');


end

