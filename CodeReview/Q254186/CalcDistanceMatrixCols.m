function [ mD ] = CalcDistanceMatrixCols( mA, mB )
% ----------------------------------------------------------------------------------------------- %
% [ mD ] = CalcDistanceMatrixRows( mA, mB )
%   Calculates the distance matrix for the input data. The distance matrix
%   is a matrix where 'mD(ii, jj) = dist(mA(:, ii), mB(:, jj));'.
%   This function uses the squared Euclidean Distance for the distance
%   metric.
% Input:
%   - mA            -   Data Matrix.
%                       Each data sample is a row of the matrix.
%                       Structure: Matrix (numVarsA x varDim).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mB            -   Data Matrix.
%                       Each data sample is a row of the matrix.
%                       Structure: Matrix (numVarsB x varDim).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - mD            -   Distance Matrix.
%                       A symmetric matrix where 'mD(ii, jj) = dist(mA(:,
%                       ii), mB(:, jj));'.
%                       Structure: Matrix (numVarsA x numVarsB).
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

mD = sum(mA .^ 2).' - (2 * mA.' * mB) + sum(mB .^ 2);


end

