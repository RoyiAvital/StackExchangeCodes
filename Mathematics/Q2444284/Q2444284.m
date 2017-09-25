% Mathematics Q2444284
% https://math.stackexchange.com/questions/2444284
% Matrix Derivative of Frobenius Norm with Hadamard Product Inside
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     25/09/2017
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

dimOrder = 10;

difMode = DIFF_MODE_CENTRAL;
epsVal  = 1e-6;


%% Generate Data

mR = randn([dimOrder, dimOrder]);
mP = randn([dimOrder, dimOrder]);


%% Validate Derivative

vX          = randn([dimOrder, 1]);
hNormFun    = @(vX) 0.5 * (norm( mP .* (vX * vX.') - mR, 'fro' ) ^ 2);

vGNumerical = CalcFunGrad(vX, hNormFun, difMode, epsVal);

mX          = vX * vX.';
vGAnalytic  = zeros([dimOrder, 1]);

for kk = 1:dimOrder
    for ii = 1:dimOrder
        vGAnalytic(kk) = vGAnalytic(kk) + mP(ii, kk) * (mP(ii, kk) .* mX(ii, kk) - mR(ii, kk)) * vX(ii);
    end
    for jj = 1:dimOrder
        vGAnalytic(kk) = vGAnalytic(kk) + mP(kk, jj) * (mP(kk, jj) .* mX(kk, jj) - mR(kk, jj)) * vX(jj);
    end
end

disp(['Maximum Deviation Between Analytic and Numerical Derivative - ', num2str( max(abs(vGNumerical - vGAnalytic)) )]);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

