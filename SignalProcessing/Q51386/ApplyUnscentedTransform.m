function [ vA, mA, mB, mAa ] = ApplyUnscentedTransform( hF, mX, vWm, vWc, outOrder )
% ----------------------------------------------------------------------------------------------- %
% [ vA, mA, mB, mAa ] = ApplyUnscentedTransform( hF, mX, vWm, vWc, outOrder )
%   Applies the Unscented Transform using the input function on a set of
%   Sigma Points. Uses the Sigma Points weights to calculate the mean and
%   covariance of the transformed Sigma Points.
% Input:
%   - hF            -   Model / Transformation Function.
%                       Structure: Function Handler.
%                       Type: Function Handler.
%                       Range: NA.
%   - mX            -   Set of Sigma Points.
%                       Each Sigma Point as the column of the matrix.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vWm           -   Mean Weights Vector.
%                       Weights used for calculation of the mean of the
%                       transformed data.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vWc           -   Covariance Weights Vector.
%                       Weights used for calculation of the covariance of 
%                       the transformed data.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - outOrder      -   Transform Output Dimension.
%                       Equals to 'size(hF(mX(:, 1), 1)'.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ...}.
% Output:
%   - vA            -   Transformed Data Mean Vector.
%                       The estimated mean of the PDF after the
%                       transformation.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mA            -   Transformed Sigma Points.
%                       Each column is the transformation of the input
%                       Sigma Points.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mB            -   Transformed Data Covariance Matrix.
%                       The estimated covariance of the PDF after the
%                       transformation.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mAa           -   Sigma Points Deviation Matrix.
%                       Each column is the deviation of the Sigma Point
%                       from the mean ('vA').
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  Unscented Transform (Wikipedia) - https://en.wikipedia.org/wiki/Unscented_transform.
%   2.  Lecture 5: Unscented Kalman Filter and General Gaussian Filtering (Simo Sarkka).
% Remarks:
%   1.  I
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     24/08/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

mA = zeros(outOrder, size(mX, 2));

for ii = 1:size(mX, 2)
    mA(:, ii) = hF(mX(:, ii));
end

vA  = mA * vWm;
mAa = mA - vA;

%{
for ii = size(mA), 2)
    mC = mC + vW(ii) * mA(:, ii) * mB(:, ii).';
end

Is equivalent of

mC = mA * diag(vW) * mB.';

Is equivalent of 

mC = (mA .* vW.') * mB.'; %<! vW is Column Vector

%}
mB = (mAa * diag(vWc) * mAa.');


end

