% Mathematics Q1492095
% https://math.stackexchange.com/questions/1492095
% Orthogonal Projection on Intersection of Convex Sets
% References:
%   1.  TFOCS - Demo: Alternating Projections - http://cvxr.com/tfocs/demos/alternating.
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     19/03/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 89; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

outOfSetThr     = 1e-6;
outOfSetCost    = 1e9;

mA = [-1, 1; 1, 0; 0, -1];
vB = [0; 2; 0];

ballRadius = 1;

numIterations   = 10;
stopThr         = 1e-6;


%% Load / Generate Data

numRows = size(mA, 1);
numCols = size(mA, 2);
numSets = numRows + 1;

vY = 2 * randn(numCols, 1);
vY = [3; 1];
vY = [4; 3];

cProjFun = cell(numSets, 1);

for ii = 1:numRows
    cProjFun{ii} = @(vY) ProjectOntoHalfSpace(vY, mA(ii, :).', vB(ii));
end

cProjFun{numSets} = @(vY) min((ballRadius / norm(vY, 2)), 1) * vY;

hObjFun = @(vX) (0.5 * sum((vX - vY) .^ 2)) + (outOfSetCost * any(((mA * vX) - vB) > outOfSetThr)) + (outOfSetCost * ((vX.' * vX) - ballRadius > outOfSetThr));


%% Solution by CVX

solverString = 'CVX';

tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numCols, 1);
    minimize( 0.5 * sum_square(vX - vY) );
    subject to
        mA * vX <= vB;
        norm(vX) <= sqrt(ballRadius);
cvx_end

toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

vXRef = vX;


%% Solution by Alternating Projections

solverString = 'Alternating Projections';

tic();

vX = AlternatingProjectionOntoConvexSets(cProjFun, vY, numIterations, stopThr);

toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The ', solverString, ' Solver Status - ', cvx_status]);
% disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

% It doesn't work
mQ(:, 1) = AlternatingProjectionOntoConvexSets(cProjFun, vY, 1, 0);
for ii = 2:numIterations
    mQ(:, ii) = AlternatingProjectionOntoConvexSets(cProjFun, mQ(:, ii - 1), 1, 0);
end


%% Solution by Dykstra's Projection Algorithm

solverString = 'Dykstra''s Projection Algorithm';

tic();
vX = OrthogonalProjectionOntoConvexSets(cProjFun, vY, numIterations, stopThr);
toc();

disp([' ']);
disp([solverString, ' Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(vX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

% It doesn't work
mW(:, 1) = OrthogonalProjectionOntoConvexSets(cProjFun, vY, 1, 0);
for ii = 2:numIterations
    mW(:, ii) = OrthogonalProjectionOntoConvexSets(cProjFun, mW(:, ii - 1), 1, 0);
end


%%

axisRadius = 5;

vXX = [-2 * axisRadius; 2 * axisRadius];
hLineObj = zeros(numRows, 1);

hFigure = figure('Position', figPosXLarge);
hAxes   = axes();
set(hAxes, 'NextPlot', 'add');
set(hAxes, 'XLim', [-axisRadius, axisRadius], 'YLim', [-axisRadius, axisRadius]);
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
for ii = 1:numRows
    if(mA(ii, 2) ~= 0)
        vYY(1) = (-mA(ii, 1) / mA(ii, 2)) * vXX(1) + vB(ii);
        vYY(2) = (-mA(ii, 1) / mA(ii, 2)) * vXX(2) + vB(ii);
        hLineObj(ii) = line(vXX, vYY);
    else
        vYY(1) = -1000;
        vYY(2) = 1000;
        hLineObj(ii) = line([vB(ii); vB(ii)], vYY);
    end
    
    set(hLineObj(ii), 'LineWidth', lineWidthNormal);
end
hRectObj = rectangle('Position', [-1, -1, 2, 2], 'Curvature', [1, 1], 'LineWidth', lineWidthNormal, 'EdgeColor', mColorOrder(2, :));
hLineObj = plot(vY(1), vY(2));
set(hLineObj, 'LineStyle', 'none', 'Marker', '*');
hLineObj = plot(vXRef(1), vXRef(2));
set(hLineObj, 'LineStyle', 'none', 'Marker', 'o');
hLineObj = plot(mQ(1, :), mQ(2, :));
set(hLineObj, 'LineStyle', 'none', 'Marker', 'x', 'MarkerSize', 12);
hLineObj = plot(mW(1, :), mW(2, :));
set(hLineObj, 'LineStyle', 'none', 'Marker', 'd', 'MarkerSize', 12);
set(get(hAxes, 'Title'), 'String', ['Orthogonal Projection onto Intersection of Convex Sets'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'x_1', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'x_2', ...
    'FontSize', fontSizeAxis);






% numSamples = 1000;
% 
% vX1 = linspace(-5, 5, numSamples);
% vX2 = linspace(-5, 5, numSamples);
% 
% mZ = zeros(numSamples);
% 
% for jj = 1:numSamples
%     for ii = 1:numSamples
%         mZ(ii, jj) = all(mA * [vX1(jj); vX2(ii)] <= vB);
%     end
% end
% 
% figure(); mesh(vX1, vX2, mZ);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

