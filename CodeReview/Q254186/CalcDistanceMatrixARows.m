function [ mD ] = CalcDistanceMatrixARows( mX )
% ----------------------------------------------------------------------------------------------- %
% [ mD ] = CalcDistanceMatrixARows( mX )
%   Calculates the distance matrix for the input data. The distance matrix
%   is a symmetric matrix where 'mD(ii, jj) = dist(mX(ii, :), mX(jj, :));'.
%   This function uses the squared Euclidean Distance for the distance
%   metric.
% Input:
%   - mX            -   Data Matrix.
%                       Each data sample is a row of the matrix.
%                       Structure: Matrix (numVars x varDim).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - mD            -   Distance Matrix.
%                       A symmetric matrix where 'mD(ii, jj) = dist(mX(ii,
%                       :), mX(jj, :));'.
%                       Structure: Matrix (numVars x numVars).
%                       Type: 'Single' / 'Double'.
%                       Range: [0, inf).
% References
%   1.  A
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     01/01/2021  Royi Avital	RoyiAvital@yahoo.com
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

vSsqX   = sum(mX .^ 2, 2);
mD      = vSsqX.'+ vSsqX - (2 * (mX * mX.'));


end

