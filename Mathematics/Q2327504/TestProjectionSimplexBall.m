% Test Projection onto Simplex Ball
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     27/06/2017  Royi Avital
%   *   First release.


%% General Parameters

run('InitScript.m');

numRows     = 5;
ballRadius  = 3;
stopThr     = 1e-6;


%% Generating Data

vY = 10 * rand([numRows, 1]) - 5;

%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vXCvx(numRows)
    minimize( norm(vXCvx - vY) )
    subject to
        sum(vXCvx) == ballRadius;
        vXCvx >= 0;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vXCvx.'), ' ]']);
disp([' ']);


%% Solution by Dual Function and Newton Iteration

vX = ProjectSimplex(vY, ballRadius, stopThr);

%{
% Analytic way to extract paramMu

numSamples  = 1000;
vParamMu    = linspace(min(vY) - ballRadius, max(vY) + ballRadius, numSamples);
vParamMu    = [min(vY) - ballRadius; sort(vY, 'ascend'); max(vY) + ballRadius];

numSamples = size(vParamMu, 1);

hObjFun = @(paramMu) sum( max(vY - paramMu, 0) ) - ballRadius;

vObjVal = zeros(numSamples, 1);
for ii = 1:numSamples
	vObjVal(ii) = hObjFun(vParamMu(ii));
end

figure();
plot(vParamMu, vObjVal);
grid('on');

if(any(vObjVal == 0))
    paramMu = vParamMu(vObjVal == 0);
else
    valX1Idx = find(vObjVal > 0, 1, 'last');
    valX2Idx = find(vObjVal < 0, 1, 'first');

    valX1 = vParamMu(valX1Idx);
    valX2 = vParamMu(valX2Idx);
    valY1 = vObjVal(valX1Idx);
    valY2 = vObjVal(valX2Idx);

    paramA = (valY2 - valY1) / (valX2 - valX1);
    paramB = valY1 - (paramA * valX1);
    paramMu = -paramB / paramA;
end

hObjFun(paramMu); %<! Should be zero

%}

disp([' ']);
disp(['Dual Function Solution Summary']);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);


%% Display Results

disp([' ']);
disp(['CVX Solution Sum - ', num2str(sum(vXCvx))]);
disp(['Dual Function Solution Sum - ', num2str(sum(vX))]);
disp(['Solutions Difference L1 Norn - ', num2str(norm(vXCvx - vX, 1))]);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

