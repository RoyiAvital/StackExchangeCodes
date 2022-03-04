function [ mY ] = DftReSample2D( mX, vSizeO )
% ----------------------------------------------------------------------------------------------- %
% [ mY ] = DFTUpSample2D( mX, vSizeO )
% Applies Sinc interpolation to the input 2D signal while preserving the
% Parseval Theorem energy.
% Input:
%   - mX                -   2D Signal Samples.
%                           The samples of the matrix (2D Signal) to be
%                           interpolated.
%                           Structure: Matrix (numRowsI x numColsI).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vSizeO            -   Size Vector of the Output Matrix.
%                           The number of samples of the output.
%                           Structure: Vector (1 x 2).
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, ...}.
% Output:
%   - mY                -   2D Output Signal Samples.
%                           The interpolated matrix samples.
%                           Equivalent to be interpolated by Sinc (Actually
%                           by Dirichlet Kernel).
%                           Structure: Matrix (numRowsO x numColsO).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  Sinc Interpolation of Discrete Periodic Signals (https://ieeexplore.ieee.org/document/388863).
%   2.  Comments on "Sinc Interpolation of Discrete Periodic Signals" (https://ieeexplore.ieee.org/document/700979).
% Remarks:
%   1.  The function takes care of the even and odd case of the samples. It
%       assumes the original signal was real hence keeps an hermitian
%       symmetry and the Parseval Theorem.
% TODO:
%   1.  Add support for downsampling.
%   Release Notes:
%   -   1.1.000     15/02/2022  Royi Avital
%       *   Supporting both upsample and downsample. 
%       *   Changed name into 'DftReSample2D()'.
%   -   1.0.000     12/02/2022  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    mX (:, :) {mustBeNumeric, mustBeReal}
    vSizeO (1, 2) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger}
end

vSizeI      = size(mX);
dataClass   = class(mX);

if(vSizeO == vSizeI)
    return;
end

mT = zeros(vSizeO(1), vSizeI(2), dataClass);
mY = zeros(vSizeO(1), vSizeO(2), dataClass);

for kk = 1:2 %<! 2 dimensions
    if(kk == 1)
        numSlices = vSizeI(2);
    else
        numSlices = vSizeO(1);
    end
    for ii = 1:numSlices
        if(kk == 1)
            mT(:, ii) = ifft(SincInterpolationDft(fft(mX(:, ii)), vSizeO(kk)), 'symmetric');
        else
            mY(ii, :) = ifft(SincInterpolationDft(fft(mT(ii, :)), vSizeO(kk)), 'symmetric');
        end
    end
end


end

