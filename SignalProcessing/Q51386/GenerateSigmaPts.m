function [ mX ] = GenerateSigmaPts( vX, mP, scalingFctr )
% ----------------------------------------------------------------------------------------------- %
% [ mX ] = GenerateSigmaPts( vX, mP, scalingFctr )
%   Generates set of Sigma Points according to the input mean and
%   covariance based on the scaling factor. The Sigma Points will have mean
%   value of 'vX' and Covariance of 'mP' when weighted with the proper
%   weights ('vWm' and 'vWc').
% Input:
%   - vX            -   Mean Vector.
%                       The mean of the input data (PDF) which the Sigma
%                       Points will be generated around. The Sigma Points
%                       will have weighted mean of 'vX'.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mP            -   Covariance Matrix.
%                       The covariance of the input data (PDF) which the 
%                       Sigma Points will be generated based on.
%                       The Sigma Points will have weighted covariance of 
%                       'mP'.
%                       Structure: Matrix.
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - scalingFctr   -   Scaling Factor.
%                       Parameter of the Generalized Unscented Transform
%                       which controls 
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
%   1.  The function generates 2n + 1 points where n is the dimension of
%       the data.
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     24/08/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

mPSqrt  = scalingFctr * chol(mP, 'lower');
mX      = [vX, vX + mPSqrt, vX - mPSqrt];


end

