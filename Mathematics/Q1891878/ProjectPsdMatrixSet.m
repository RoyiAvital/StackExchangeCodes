function [ mX ] = ProjectPsdMatrixSet( mX )
% ----------------------------------------------------------------------------------------------- %
% [ mX ] = ProjectPsdMatrixSet( mX )
%   Projecting the input symmetric matrix onto the Convex Set of Positive
%   Semidefinite (PSD) Matrices.
% Input:
%   - mX            -   Input Matrix.
%                       Symmetric matrix.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - mX            -   Output Matrix.
%                       Symmteirc Matrix which is the orthogonal projection
%                       of the input matrix onto the PSD Matrices Set.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  h
% Remarks:
%   1.  The input matrix is assumed to be Symmetric.
%   2.  If one wants to project onto the PD set one should send (mX - eps()
%       * mI).
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     15/08/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

[mV, mD] = eig(mX);
vD = diag(mD);
vD = max(vD, 0); %<! Thresholding

mX = mV * diag(vD) * mV.';


end

