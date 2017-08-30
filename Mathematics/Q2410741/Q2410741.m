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

numRows = 2;

paramLambda = 0.1 * rand([1, 1]);

difMode = DIFF_MODE_FORWARD;
epsVal  = 1e-6;

numSamples      = 1000;
numIterations   = 2500;
stepSize        = 0.001;


%% Generate Data

vX = rand([numRows, 1]);

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


%% Minimization

mZ      = zeros([numSamples, numSamples]);
vXGrid  = linspace(-5, 5, numSamples);

for jj = 1:numSamples
    for ii = 1:numSamples
        mZ(ii, jj) = hObjFun([vXGrid(jj); vXGrid(ii)]);
    end
end

[minVal, argMin] = min(mZ(:));
[minValRowIdx, minValColIdx] = ind2sub([numSamples, numSamples], argMin);

% Numercial Optimization

mX          = zeros([numRows, numIterations]);
vObjFunVal  = zeros([numIterations, 1]);

% Initialization
vX              = rand([numRows, 1]);
vX              = [-2; 4];

mX(:, 1)        = vX;
vObjFunVal(1)   = hObjFun(vX);

for ii = 2:numIterations
    vG              = CalcFunGrad(vX, hObjFun, difMode, epsVal);
    vX              = vX - (stepSize * vG);
    mX(:, ii)       = vX;
    vObjFunVal(ii)  = hObjFun(vX);
end

figureIdx = figureIdx + 1;

hFigure         = figure('Position', [100, 100, 720, 960]);
hAxes           = subplot(3, 1, 1:2);
set(hAxes, 'NextPlot', 'add');
hImageObject    = image('XData', vXGrid, 'YData', vXGrid, 'CData', mZ, 'CDataMapping', 'scaled');
[mCountourLines, hContouObj]  = contour(vXGrid, vXGrid, mZ);
set(hContouObj, 'LineColor', 'white');
hLineSeries(1)  = line(vXGrid(minValColIdx), vXGrid(minValRowIdx));
set(hLineSeries(1), 'LineStyle', 'none', 'Marker', 'diamond', 'MarkerSize', markerSizeLarge, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r');
hLineSeries(2)  = line(mX(2, :), mX(1, :));
set(hLineSeries(2), 'LineStyle', 'none', 'Marker', 'o', 'MarkerSize', 5, 'MarkerEdgeColor', 'm', 'MarkerFaceColor', 'w');
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
set(get(hAxes, 'Title'), 'String', {['Objective Function Value']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['x_2']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['x_1']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend([hLineSeries(1), hLineSeries(2)], {['Optimal Argument'], ['Gradient Descent Iterations']});

hAxes = subplot(3, 1, 3);
hLineSeries = plot(1:numIterations, vObjFunVal);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Objective Function Value']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Iteration Number']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
    'FontSize', fontSizeAxis);


saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);




%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

