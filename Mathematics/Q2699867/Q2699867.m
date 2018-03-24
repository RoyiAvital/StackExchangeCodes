% Mathematics Q2699867
% https://math.stackexchange.com/questions/2699867
% Minimizing quadratic form with norm and positive orthant constraints
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     24/03/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numRows = 2;
numCols = numRows;

numIterations   = 1000;
stepSize        = 0.00075;


%% Generate Data

mA = randn([numRows, numCols]);
mM = (mA.' * mA) + (0.05 * eye(numRows));

hObjFun = @(vX) vX.' * mM * vX;
hProjL2Fun = @(vX) vX ./ norm(vX, 2);
hProjRPlusFun = @(vX) max(vX, 0);


%% Solution by Projected Gradient Method

vObjValPgd = zeros([numIterations, 1]);

vX = hProjL2Fun(ones([numRows, 1]));
mX = zeros([numRows, numIterations]);
mX(:, 1) = vX;

vObjValPgd(1) = hObjFun(vX);

for ii = 2:numIterations
    vG = 2 * mM * vX;
    vX = vX - (stepSize * vG);
    vX = hProjRPlusFun(vX);
    vX = hProjL2Fun(vX);
    
    mX(:, ii) = vX;
    
    vObjValPgd(ii) = hObjFun(vX); 
end

disp([' ']);
disp(['Projected Gradient Descent Method Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjValPgd(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX(:).'), ' ]']);
disp([' ']);


%% Display Reesults

gridNumSamples = 1000;

minVal = inf;
vMinValCoord = [0; 0];

vGrid   = linspace(-2, 2, gridNumSamples);
mObjVal = zeros([gridNumSamples, gridNumSamples]);

for ii = 1:gridNumSamples %<! {x}_{1}
    for jj = 1:gridNumSamples %<! {x}_{2}
        vG = [vGrid(ii); vGrid(jj)];
        mObjVal(jj, ii) = hObjFun(vG);
        if((abs(norm(vG, 2) - 1) < 1e-4) && (all(vG >= 0)))
            if(mObjVal(jj, ii) < minVal)
                minVal = mObjVal(jj, ii);
                vMinValCoord = vG;
            end
        end
    end
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100, 100, 720, 720]);
hAxes       = axes();
set(hAxes, 'NextPlot', 'add');
hImageObj = imagesc(vGrid, vGrid, mObjVal);
set(hAxes, 'XDir', 'normal', 'YDir', 'normal');
set(hAxes, 'DataAspectRatio', [1, 1, 1]);

hRectObj = rectangle(hAxes, 'Position', [-1, -1, 2, 2], 'Curvature', [1, 1]);
set(hRectObj, 'LineWidth', lineWidthNormal);

hLineObj = line(mX(1, :), mX(2, :));
set(hLineObj, 'LineStyle', 'none', 'Marker', '*', 'MarkerSize', markerSizeNormal, 'Color', 'r');

hLineObj = line(vMinValCoord(1), vMinValCoord(2));
set(hLineObj, 'LineStyle', 'none', 'Marker', '*', 'MarkerSize', markerSizeNormal, 'Color', 'g');

set(get(hAxes, 'Title'), 'String', ['Objective Function & Projected Gradient Descend Path'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', '{x}_{1}', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', '{x}_{2}', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Projected Gradient Descend'], ['Optimal Solution']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

