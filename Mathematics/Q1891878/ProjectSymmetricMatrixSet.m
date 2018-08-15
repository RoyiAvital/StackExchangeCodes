function [ mX ] = ProjectSymmetricMatrixSet( mX )
% ----------------------------------------------------------------------------------------------- %
% [ mX ] = ProjectSymmetricMatrixSet( mX )
%   Projecting the input matrix into the Convex Set of Symmetric Matrices.
% Input:
%   - mX            -   Input Matrix.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - mX            -   Output Matrix.
%                       Symmteirc Matrix which is the orthogonal projection
%                       of the input matrix.
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
%   -   1.0.000     15/08/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

mX = (mX.' + mX) / 2;


end

