% Mathematics Q2246047
% https://math.stackexchange.com/questions/2246047
% Projection onto Positive Semidefinite (PSD) Matrices with Bounded Rank
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     31/05/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numRows         = 8;

stepSize        = 1e-1;
numIterations   = 25000;

rankK   = 3; %<! Rank K


%% Generate Data

mA = randn([numRows, numRows]);


%% Solution by Projected Sub Gradient

hPsdMat     = @(mX) 1e9 * any(abs(eig(mX) < -1e-9));
hRankKMAt   = @(mX) 1e9 * (rank(mX) > rankK);


hObjFun     = @(mX) 0.5 * sum((mX(:) - mA(:)) .^ 2) + hPsdMat(mX) + hRankKMAt(mX);
hProjFun    = @(mX) ProjectPsdRankK(mX, rankK);

vObjVal = zeros([numIterations, 1]);

% First Iteration
mX          = hProjFun(mA);
vObjVal(1)  = hObjFun(mX);

for ii = 2:numIterations
    
    mG = mX - mA;
    mX = mX - (stepSize * mG);
    mX = hProjFun(mX);
    
    vObjVal(ii) = hObjFun(mX);
end

% disp([' ']);
% disp(['Projected Sub Gradient Solution Summary']);
% disp(['The Optimal Value Is Given By - ', num2str(vObjVal(numIterations))]);
% disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
% disp([' ']);

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot(1:numIterations, vObjVal);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['Objective Function Value vs. Iteration'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
set(hAxes, 'XLim', [1, numIterations]);
% hLegend = ClickableLegend({['Projected Sub Gradient'], ['Optimal Value (CVX)']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

