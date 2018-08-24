function [ vWm, vWc, scalingFactor ] = CalcSigmaPointsWeights( paramAlpha, paramBeta, paramKappa, stateOrder )
% ----------------------------------------------------------------------------------------------- %
% [ vWm, vWc, scalingFactor ] = CalcSigmaPointsWeights( paramAlpha, paramBeta, paramKappa, stateOrder )
%   Calculates the weights for the mean, covariance and scaling sactor of 
%   the Unsecented Transform.
% Input:
%   - paramAlpha    -   Parameter Alpha.
%                       Influences how far the Sigma Points are form the
%                       mean.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, 1].
%   - paramBeta     -   Parameter Beta.
%                       For Gaussian PDF the optimal value is 2.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: NA.
%   - paramKappa    -   Parameter Kappa.
%                       Influences how far the Sigma Points are form the
%                       mean.
%                       Type: 'Single' / 'Double'.
%                       Range: [0, inf).
%   - stateOrder    -   State Vector  Dimension.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ...}.
% Output:
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
%   - scalingFctr   -   Scaling Factor.
%                       Parameter of the Generalized Unscented Transform
%                       which controlls 
%                       Equals to 'size(hF(mX(:, 1), 1)'.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ...}.
% References
%   1.  Unscented Tranform (Wikipedia) - https://en.wikipedia.org/wiki/Unscented_transform.
%   2.  Lecture 5: Unscented Kalman Filter and General Gaussian Filtering (Simo Sarkka).
%   3.  Robot Mapping - Unscented Kalman Filter (Cyrill Stachniss).
% Remarks:
%   1.  This method of parameters is often called Generalized / Scaled
%       Unscented Kalman Filter.
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     24/08/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

numSigmaPoints  = (2 * stateOrder) + 1;

vWm = zeros(numSigmaPoints, 1);
vWc = zeros(numSigmaPoints, 1);

paramLambda = ((paramAlpha * paramAlpha) * (stateOrder + paramKappa)) - stateOrder;

vWm(1)      = paramLambda / (stateOrder + paramLambda);
vWm(2:end)  = 1 / (2 * (stateOrder + paramLambda));
vWc(1)      = vWm(1) + (1 - (paramAlpha * paramAlpha) + paramBeta);
vWc(2:end)  = vWm(2:end);

scalingFactor = sqrt(stateOrder + paramLambda);


end

