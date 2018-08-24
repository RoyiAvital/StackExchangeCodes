function [ vTransVectorMean, vTransVectorCov ] = UnscentedTransform( hTransformFunction, vVectorMean, mVectorCov )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

vectorOrder     = size(vVectorMean, 1);
numSigmaPoints  = (2 * vectorOrder) + 1;

mVectorCovSqrt = chol(mVectorCov);
if(vectorOrder <= 3) %<! Used as k Factor in literature
    unscentedTransformScalingFactor = 3 - vectorOrder;
else
    unscentedTransformScalingFactor = 3;
end


vSigmaPoints                = zeros(vectorOrder, numSigmaPoints);
vTransformedSigmaPoints     = zeros(vectorOrder, numSigmaPoints);
vUnscentedTransformWeights  = zeros(numSigmaPoints, 1);

vTransVectorMean    = zeros(vectorOrder, 1);
vTransVectorCov     = zeros(vectorOrder, vectorOrder);

mVectorCovSqrtNormalized = sqrt(vectorOrder + unscentedTransformScalingFactor) .* mVectorCovSqrt;

vSigmaPoints(1) = 0;
vUnscentedTransformWeights(1) = unscentedTransformScalingFactor / (unscentedTransformScalingFactor + vectorOrder);
vUnscentedTransformWeights(2:end) = 1 / (2 * (unscentedTransformScalingFactor + vectorOrder));

for ii = 2:(vectorOrder + 1)
    vSigmaPoints(:, ii) = mVectorCovSqrtNormalized(:, (ii - 1));
end

vSigmaPoints(:, (vectorOrder + 2):end) = -vSigmaPoints(:, 2:(vectorOrder + 1));

for ii = 1:numSigmaPoints
    vSigmaPoints(:, ii) = vVectorMean + vSigmaPoints(:, ii);
    vTransformedSigmaPoints(:, ii) = hTransformFunction(vSigmaPoints(:, ii));
end


for ii = 1:numSigmaPoints
    vTransVectorMean = vTransVectorMean + (vUnscentedTransformWeights(ii) * vTransformedSigmaPoints(:, ii));
end



for ii = 1:numSigmaPoints
    vTransVectorCov = vTransVectorCov + ...
        (vUnscentedTransformWeights(ii) * (([vTransformedSigmaPoints(:, ii) - vTransVectorMean]) * ([vTransformedSigmaPoints(:, ii) - vTransVectorMean].')));
end


end

