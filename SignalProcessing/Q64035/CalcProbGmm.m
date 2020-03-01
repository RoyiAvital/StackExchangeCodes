function [ vP ] = CalcProbGmm( vMu, vSigma, vModelProb, vX )
% ----------------------------------------------------------------------------------------------- %

numPts = size(vX, 1);
vP = zeros(numPts, 1);

numModels = size(vMu, 1);

for jj = 1:numPts
    valX = vX(jj);
    probVal = 0;
    for ii = 1:numModels
        normFactor  = 1 / (vSigma(ii) * sqrt(2 * pi)); %<! Normalization Factor
        devVal      = valX - vMu(ii); %<! Deviation from the mean
        probVal     = probVal + (vModelProb(ii) * normFactor * exp(-(devVal * devVal) / (2 * vSigma(ii) * vSigma(ii))));
    end
    vP(jj) = probVal;
end

end

