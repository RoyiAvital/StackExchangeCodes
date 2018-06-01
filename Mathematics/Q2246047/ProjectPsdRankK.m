function [ mX ] = ProjectPsdRankK( mY, rankK )
% ----------------------------------------------------------------------------------------------- %
% [ mX ] = ProjectPsdRankK( mY, rankK )
%   Projeting onto the PSD with Rank K matrices.
% Input:
%   - mY            -   Input Matrix.
%                       The matrix to be projected.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - rankK         -   The Rank 'k;.
%                       Sets the Rank of the output matrix.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
% Output:
%   - mX            -   Output Matrix.
%                       A PSD matrix with rank 'k'.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  h
% Remarks:
%   1.  a
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     31/05/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

% Symmetrize (Doing both
mY = (mY.' + mY) / 2;
mY = (mY + mY.') / 2;

[mU, mS, mV] = svd(mY);

[mV, mD] = eig(mY);

vD = diag(mD); %<! 'eig' output is sorted in ascending order

vD = max(vD, 0); %!< Thresholding
vD(1:end - rankK) = 0; %<! Rank

mX = mV * diag(vD) * mV.';


end

