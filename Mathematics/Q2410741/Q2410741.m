% Mathematics Q2410741
% https://math.stackexchange.com/questions/2410741
% Matrix differentiation on ∥A−B∘X∥2F+λ∥X∥2F
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     30/08/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;


%% Simulation Parameters

numRows = 5;

paramLambda = rand([1, 1]);
% paramLambda = 0;

difMode = DIFF_MODE_FORWARD;
epsVal  = 1e-6;


%% Generate Data

vX = randn([numRows, 1]);

mA = randn([numRows, numRows]);
mB = randn([numRows, numRows]);

hObjFun = @(vX) (norm(mA - (mB .* (vX * vX.')), 'fro') .^ 2) + (paramLambda * (norm(vX * vX.', 'fro') ^ 2));
% hObjFun = @(vX) sum(sum((mA - mB .* (vX * vX.')) .^ 2)) + (paramLambda * sum(sum((vX * vX.') .^ 2)));


%% Numerical Solution

vGNumerical = CalcFunGrad(vX, hObjFun, difMode, epsVal);


%% Analytic Solution

vG = zeros([numRows, 1]);


for kk = 1:numRows
    for ii = 1:numRows
        for jj = 1:numRows
            if(ii == kk)
                vG(kk) = vG(kk) + (-2 * mB(ii, jj) * vX(jj) * (mA(ii, jj) - mB(ii, jj) * vX(ii) * vX(jj)));
                vG(kk) = vG(kk) + (2 * paramLambda * vX(jj) * (vX(ii) * vX(jj)));
            end
            if(jj == kk)
                vG(kk) = vG(kk) + (-2 * mB(ii, jj) * vX(ii) * (mA(ii, jj) - mB(ii, jj) * vX(ii) * vX(jj)));
                vG(kk) = vG(kk) + (2 * paramLambda * vX(ii) * (vX(ii) * vX(jj)));
            end
        end
    end
end


%% Analysis

disp(['Maximum Error Between Numercial Differntiation to Analytic - ', num2str(max(abs(vG - vGNumerical)))]);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

