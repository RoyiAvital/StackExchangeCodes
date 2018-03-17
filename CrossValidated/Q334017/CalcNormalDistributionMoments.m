function [ distMoment ] = CalcNormalDistributionMoments( paramMu, paramSigmaSquared, momentOrder )
% See https://en.wikipedia.org/wiki/Normal_distribution#Moments.

switch(momentOrder)
    case(1)
        distMoment = (paramMu ^ momentOrder);
    case(2)
        distMoment = (paramMu ^ momentOrder) + paramSigmaSquared;
    case(3)
        distMoment = (paramMu ^ momentOrder) + (3 * paramMu * paramSigmaSquared);
    case(4)
        distMoment = (paramMu ^ momentOrder) + (6 * (paramMu ^ 2) * paramSigmaSquared) + (3 * (paramSigmaSquared ^ 2));
    case(5)
        distMoment = (paramMu ^ momentOrder) + (10 * (paramMu ^ 3) * paramSigmaSquared) + (15 * paramMu * (paramSigmaSquared ^ 2));
    case(6)
        distMoment = (paramMu ^ momentOrder) + (15 * (paramMu ^ 4) * paramSigmaSquared) + (45 * (paramMu ^ 2) * (paramSigmaSquared ^ 2)) + (15 * (paramSigmaSquared ^ 3));
    case(7)
        distMoment = (paramMu ^ momentOrder) + (21 * (paramMu ^ 5) * paramSigmaSquared) + (105 * (paramMu ^ 3) * (paramSigmaSquared ^ 2)) + (105 * paramMu * (paramSigmaSquared ^ 3));
    case(8)
        distMoment = (paramMu ^ momentOrder) + (28 * (paramMu ^ 6) * paramSigmaSquared) + (210 * (paramMu ^ 4) * (paramSigmaSquared ^ 2)) + (420 * (paramMu ^ 2) * (paramSigmaSquared ^ 3)) + (105 * (paramSigmaSquared ^ 4));
    otherwise
        gridRadius      = 10 * sqrt(paramSigmaSquared);
        gridNaumSamles  = 1e6;
        vX = linspace(paramMu - gridRadius, paramMu + gridRadius, gridNaumSamles);
        dX = mean(diff(vX));
        vNormalPdf = (1 / sqrt(2 * pi * paramSigmaSquared)) * exp(-( (vX - paramMu) .^ 2 ) ./ (2 * paramSigmaSquared));
        distMoment = sum( (vX .^ momentOrder) .* vNormalPdf * dX);
end


end

