
%% Numerical Gradient Settings

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;

epsVal  = 1e-6;
difMode = DIFF_MODE_FORWARD;


%% Gradient of (1 / 2) * || vA.' * vX * vB || ^ 2

dimA = 5;
dimB = 3;

vA = randn([dimA, 1]);
vX = randn([dimA, 1]);
vB = randn([dimB, 1]);

hObjFun     = @(vX) 0.5 * sum((vA.' * vX * vB) .^ 2);
hGradFun1   = @(vX) vA * vB.' * (vA.' * vX) * vB;
hGradFun2   = @(vX) vB.' * vB * vA.' * vX * vA;

vG1 = hGradFun1(vX); %<! By hand
vG2 = hGradFun2(vX); %<! From http://www.matrixcalculus.org/
vG3 = CalcFunGrad(vX, hObjFun, difMode, epsVal); %<! Numerically

[vG1, vG2, vG3]


%% Gradient of (1 / 2) * || mX vW - (1 / n) vE.' * mX * vW * vE || ^ 2

dimOrder    = 3;
numSamples  = 4;

mX = randi([1, 10], [dimOrder, numSamples]);
vE = ones([dimOrder, 1]);

hKernelFun  = @(vW) ((mX * vW) - ((1 / numSamples) * ((vE.' * mX * vW) * vE)));
hObjFun     = @(vW) 0.5 * sum(hKernelFun(vW) .^ 2);
% hGradFun1   = @(vW) (mX.' * hKernelFun(vW)) - ((1 / numSamples) * (mX.' * (vE * vE.')) * hKernelFun(vW));
hGradFun1   = @(vW) (mX.' - ( (1 / numSamples) * mX.' * (vE * vE.') )) * hKernelFun(vW); %<! Same as above
hGradFun2   = @(vW) (mX.' * hKernelFun(vW)) - ((1 / numSamples) * vE.' * (hKernelFun(vW)) * mX.' * vE);

vW = rand([numSamples, 1]);
vW = vW(:) / sum(vW);

vG1 = hGradFun1(vW); %<! By Hand
vG2 = hGradFun2(vW); %<! From http://www.matrixcalculus.org/
vG3 = CalcFunGrad(vW, hObjFun, difMode, epsVal); %<! Numerically

[vG1, vG2, vG3]

