function [ paramAlpha, paramBeta ] = EstimateModelParameters( vT, vY )

vParamAlpha = [1, 2, 3];
vParamBeta  = [1, 2, 3];

bestMse = 1e50;

vX = zeros(size(vY, 1), 1, class(vY));

for ii = 1:length(vParamAlpha)
    currParamAlpha = vParamAlpha(ii);
    for jj = 1:length(vParamBeta)
        currParamBeta = vParamBeta(jj);
        vX(:) = currParamAlpha * (vT .^ currParamBeta);
        currMse = mean((vX - vY) .^ 2);
        if(currMse < bestMse)
            bestMse     = currMse;
            paramAlpha  = currParamAlpha;
            paramBeta   = currParamBeta;
        end
    end
end


end


