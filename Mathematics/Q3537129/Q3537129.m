% StackExchange Mathematics Q3537129
% https://math.stackexchange.com/questions/3537129
% Projection of z onto the Affine Half Space {x∣Ax=b,x≥0}.
% References:
%   1.  
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     27/12/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 2088;0; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

numRows         = 10;
numCols         = 20;

% Solvers parameters
numIterations   = 500;
stopThr         = 1e-6;


%% Load / Generate Data

mA = randn(numRows, numCols);
vB = randn(numRows, 1);
vZ = randn(numCols, 1);

mAA     = mA * mA.';
mInvAA  = pinv(mAA);

hObjFun = @(vX) 0.5 * sum((vX - vZ) .^ 2);

hP1 = @(vX) vX - (mA.' * mInvAA * ((mA * vX) - vB)); %<! Projection onto the 1st constraint set
hP2 = @(vX) max(vX, 0); %<! Projection onto the 2nd constraint set

hP = @(vX) OrthogonalProjectionOntoConvexSets({hP1; hP2}, vX, numIterations, stopThr);

solverIdx       = 0;
cMethodString   = {};


%% Solution by CVX

solverString = 'CVX';

% cvx_solver('SDPT3'); %<! Default, Keep numRows low
% cvx_solver('SeDuMi');
% cvx_solver('Mosek'); %<! Can handle numRows > 500, Very Good!
% cvx_solver('Gurobi');

hRunTime = tic();

cvx_begin('quiet')
% cvx_begin()
    % cvx_precision('best');
    variable vX(numCols, 1);
    minimize( 0.5 * sum_square(vX - vZ) );
    subject to
        mA * vX == vB;
        vX >= 0;
cvx_end

runTime = toc(hRunTime);

% vX = mX(:);

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);

sCvxSol.vXCvx     = vX;
sCvxSol.cvxOptVal = hObjFun(vX);


%% Solution by Dykstra's Projection Algorithm

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Projected Gradient Method'];

hRunTime = tic();

vX = hP(vZ);

runTime = toc(hRunTime);

disp([' ']);
disp([cLegendString{solverIdx}, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

