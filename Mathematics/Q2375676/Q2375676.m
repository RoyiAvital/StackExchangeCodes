% Mathematics Q2375676
% https://math.stackexchange.com/questions/2375676
% Minimize the Squared L2 Norm of a Vector With Linear Equality and Inequality Constraints
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     14/02/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 0; %<! 2115

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numElements = 8;


%% Generate Data

vA          = 10 * rand(numElements, 1);
paramB      = 7 * rand(1);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numElements);
    minimize( pow_pos(norm(vX, 2), 2) );
    subject to
        vX - vA <= 0;
        sum(vX) == paramB;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp([' ']);


%% Solution by KKT
%{
Solving using KKT - See https://math.stackexchange.com/a/2375679.
%}

[vAA, vAIdx] = sort(vA, 'ascend');

vSetA = true(numElements, 1);
vSetB = false(numElements, 1);

vXX = (paramB / sum(vSetA)) * ones(numElements, 1);

for ii = 1:numElements
    
    if(ii == 1)
        if(all(vXX <= vAA))
            break;
        end
    else
        if( all(((paramB - sum(vXX(vSetB))) / sum(vSetA)) <= vAA(vSetA)) )
            break;
        end
    end
    
    vSetA(ii) = false;
    vSetB(ii) = true;
end

vXX(vSetB) = vAA(vSetB);
vXX(vSetA) = (paramB - sum(vXX(vSetB))) / sum(vSetA);

% Sorting it according to the actual vA
vX = zeros(numElements, 1);

for ii = 1:numElements
    vX(vAIdx(ii)) = vXX(ii);
end


disp([' ']);
disp(['KKT Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str( vX.' * vX )]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp([' ']);


%% Solution by Dual Problem (Non Negative Least Squares / Quadratic Programming)
%{
Solving using the Dual Problem - See https://math.stackexchange.com/a/2375690.
%}

mE = [eye(numElements), ones(numElements, 1)];
mH = 0.5 * (mE.' * mE);

mA = - [eye(numElements), zeros(numElements, 1)];
vF = [vA; paramB];
vB = zeros(numElements, 1);

vLambda = quadprog(mH, vF, mA, vB);
vX = -0.5 * (mE * vLambda);

disp([' ']);
disp(['Quadratic Programming Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str( vX.' * vX )]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp([' ']);


%% Display Results

figureIdx = figureIdx + 1;

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

