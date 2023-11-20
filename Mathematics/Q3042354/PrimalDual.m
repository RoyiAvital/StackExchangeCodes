function [ vX, mX ] = PrimalDual( vX, mX, vP, mA, hProxFS, hProxG, paramTheta, numIterations )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

%   Solves 
%       min_x f(A * x) + g(x)
% https://link.springer.com/article/10.1007/s10851-010-0251-1
% http://maths.nju.edu.cn/~hebma/Talk/Simple_Power.pdf (Page 21).

valL        = normest(mA.' * mA);
paramSigma  = 10;
paramTau    = 0.9 / (paramSigma * valL);
paramSigma  = sqrt(1 / (1.05 * valL));
paramTau    = sqrt(1 / (1.05 * valL));

vXX = vX;

for ii = 2:numIterations
    vXPrev = vX;
    vP(:) = hProxFS(vP + paramSigma * mA * vXX, paramSigma);
    vX(:) = hProxG(vX - paramTau * mA' * vP, paramTau);

    vXX(:) = vX + paramTheta * (vX - vXPrev);
    
    mX(:, ii) = vX;
end


end

