% Mathematics Q1894063
% https://math.stackexchange.com/questions/1894063
% Proximal Operator for $ f \left( x \right) = \sqrt{ {x}^{2} + {a}^{2} } $
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     03/08/2019
%   *   First release.


%% General Parameters

% Set to 0 for random SubStream
subStreamNumberDefault = 0;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;
DIFF_MODE_COMPLEX   = 4;


%% Parameters

paramLambda = 5 * rand(1);
valX = randn(1);
valY = 3 * randn(1);
paramA = 5 * rand(1);

diffMode = DIFF_MODE_COMPLEX;
epsVal = 1e-6;

numIterations   = 1e6;
stepSize        = 0.0005;


%% Load / Generate Data

hProxFun = @(valX) 0.5 * ((valX - valY) ^ 2) + (paramLambda * sqrt((valX ^ 2) + (paramA ^ 2)));


%% The Sub Gradient

% Numerical Gradient
valG = CalcFunGrad(valX, hProxFun, diffMode, epsVal)
% Analytic Gradient
valG = (valX - valY) + (paramLambda * (valX) / (sqrt((valX ^ 2) + (paramA ^ 2))))


%% Optimization

% Solution by CVX

% tic();
% 
% cvx_begin('quiet')
%     % cvx_precision('best');
%     variable valX(1);
%     minimize( (0.5 * sum_square(valX - valY)) + (paramLambda * norm([valX; paramA])) );
% cvx_end
% 
% toc();
% 
% disp([' ']);
% disp(['CVX Solution Summary']);
% disp(['The CVX Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
% disp([' ']);

% Solution by Gradient Descent

valX = valY;

for ii = 1:numIterations
    valG = (valX - valY) + (paramLambda * (valX) / (sqrt((valX ^ 2) + (paramA ^ 2))));
    valX = valX - (stepSize * valG);
end

disp(['valX = ', num2str(valX)]);
disp(['Prox Function Value = ', num2str(hProxFun(valX))]);

valXRef = valX;


% Analytic Solution
% Quartic Function Coefficients
vC = [1, -2 * valY, ((paramA  ^ 2) + (valY ^ 2) - (paramLambda ^ 2)), -2 * (paramA ^ 2) * valY, (paramA ^ 2) * (valY ^ 2)];
% Polynomial Roots
vR = roots(vC);

vRealIdx = find(imag(vR) == 0);

minVal = inf;
solIdx = 0;
for ii = 1:length(vRealIdx)
    proxVal = hProxFun(vR(vRealIdx(ii)));
    if(proxVal < minVal)
        solIdx = vRealIdx(ii);
        minVal = proxVal;
    end
end

valX = vR(solIdx);

disp(['valX = ', num2str(valX)]);
disp(['Prox Function Value = ', num2str(hProxFun(valX))]);

disp(['abs(valXRef - valX) = ', num2str(abs(valXRef - valX))]);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

