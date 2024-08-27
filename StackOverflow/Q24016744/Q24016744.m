% StackOverflow Q24016744
% https://stackoverflow.com/questions/24016744
% Creating Filter's Laplacian Matrix and Solving the Linear Equation for Image Filtering
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     25/08/2024
%   *   First release.


%% General Parameters

subStreamNumberDefault = 0;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants



%% Parameters

% Data
% From Image Smoothing via L0 Gradient Minimization (https://www.cse.cuhk.edu.hk/~leojia/projects/L0smoothing/ImageSmoothing.htm)
imgUrl = 'https://i.imgur.com/8LuDAju.png'; %<! Basketball
% imgUrl = 'https://i.imgur.com/LzBNqV8.png'; %<! Beach
% imgUrl = 'https://i.imgur.com/69CLVcn.png'; %<! Flower
% imgUrl = 'https://i.imgur.com/r9nUFsM.png'; %<! Mountain

% Model
paramLambda = 2.15;
paramGamma  = 1.5;
paramEps    = 1e-4;

% Solver
numIter = 5000;
solThr  = 1e-6;


%% Generate / Load Data

mI = im2double(imread(imgUrl));
mI = mean(mI, 3);

numRows = size(mI, 1);
numCols = size(mI, 2);

numPx = numRows * numCols;

% mI = rand(numRows, numCols);

vY = mI(:);


%% Analysis

mA = GenMatA(vY, numRows, numCols, paramLambda, paramGamma, paramEps);

[mAx, mAy] = GenMatAxAy(vY, numRows, numCols, paramGamma, paramEps);


hOpA = @(vT) ApplyOpA(vT, numRows, numCols, mAx, mAy, paramLambda);

% Applying operators
vT1 = mA * vY;
vT2 = hOpA(vY);

max(abs(vT1 - vT2), [], 'all')

for ii = 1:2
    vT1 = mA * vT1;
    vT2 = hOpA(vT2);
end

max(abs(vT1 - vT2), [], 'all')

% hF = @() mA * vY;
hF = @() GenMatA(vY, numRows, numCols, paramLambda, paramGamma, paramEps) * vY;
hG = @() hOpA(vY);

TimeItMin(hF, 1)
TimeItMin(hG, 1)

condest(mA)

% Direct Solution
vXD = mA \ vY;
mL = ichol(mA, struct('michol', 'on'));
vX = pcg(hOpA, vY, solThr, numIter);
% vX = pcg(hOpA, vY, solThr, numIter, mL, mL.');
% vX = pcg(mA, vY, solThr, numIter);
% vX = pcg(mA, vY, solThr, numIter, mL, mL.');

hF = @() mA \ vY;
hG = @() pcg(hOpA, vY, solThr, numIter);

TimeItMin(hF, 1)
TimeItMin(hG, 1)


figure();
imshow(reshape(vY, numRows, numCols), [0, 1]);

figure();
imshow(reshape(vXD, numRows, numCols), [0, 1]);

figure();
imshow(reshape(vX, numRows, numCols), [0, 1]);

sqrt(mean(abs(vX - vXD) .^ 2))



%% Display Results

% figureIdx = figureIdx + 1;
% 
% hFigure = figure('Position', figPosLarge);
% hAxes   = axes(hFigure);
% set(hAxes, 'NextPlot', 'add');
% hLineObj = plot(mX);
% for ii = 1:numSignals
%     set(hLineObj(ii), 'DisplayName', ['Line ', num2str(ii)]);
% end
% set(hLineObj, 'LineWidth', lineWidthNormal);
% hLineObj = plot(vX, 'DisplayName', 'Mean Line');
% set(hLineObj, 'LineWidth', lineWidthNormal, 'LineStyle', ':');
% 
% set(get(hAxes, 'Title'), 'String', {['Estimate Data by Partial Data']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
%     'FontSize', fontSizeAxis);
% 
% hLegend = ClickableLegend();
% 
% if(generateFigures == ON)
%     % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
%     print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
% end
% 
% 
% figureIdx = figureIdx + 1;
% 
% hFigure = figure('Position', figPosLarge);
% hAxes   = axes(hFigure);
% set(hAxes, 'NextPlot', 'add');
% hLineObj = plot(mX);
% for ii = 1:numSignals
%     set(hLineObj(ii), 'DisplayName', ['Line ', num2str(ii)]);
% end
% set(hLineObj, 'LineWidth', lineWidthNormal);
% hLineObj = plot(vXX, 'DisplayName', 'Optimized Line');
% set(hLineObj, 'LineWidth', lineWidthNormal, 'LineStyle', ':');
% 
% set(get(hAxes, 'Title'), 'String', {['Estimate Data by Partial Data']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
%     'FontSize', fontSizeAxis);
% 
% hLegend = ClickableLegend();
% 
% if(generateFigures == ON)
%     % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
%     print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
% end


%% Auxiliary Functions

function [ mA ] = GenMatA( vY, numRows, numCols, paramLambda, paramGamma, paramEps )

numElements = numRows * numCols;

mDx = CreateConvMtx2D([1, -1], numRows, numCols, 3); %<! 3 -> Conv Shape Valid
mDy = CreateConvMtx2D([1; -1], numRows, numCols, 3); %<! 3 -> Conv Shape Valid

% Could apply λ in this step
mAx = spdiags(1 ./ ((abs(mDx * vY) .^ paramGamma) + paramEps), 0, numElements - numRows, numElements - numRows);
mAy = spdiags(1 ./ ((abs(mDy * vY) .^ paramGamma) + paramEps), 0, numElements - numCols, numElements - numCols);
    
mA = speye(numElements) + (paramLambda * (mDy.' * mAy * mDy + mDx.' * mAx * mDx));

end

function [ mAx, mAy ] = GenMatAxAy( vY, numRows, numCols, paramGamma, paramEps )

mY = reshape(vY, numRows, numCols);
mIx = conv2(mY, [1, -1], 'valid');
mIy = conv2(mY, [1; -1], 'valid');

mAx = 1 ./ ((abs(mIx) .^ paramGamma) + paramEps);
mAy = 1 ./ ((abs(mIy) .^ paramGamma) + paramEps);


end


function [ vX ] = ApplyOpA( vY, numRows, numCols, mAx, mAy, paramLambda )

mY = reshape(vY, numRows, numCols);
mIx = conv2(mY, [1, -1], 'valid');
mIy = conv2(mY, [1; -1], 'valid');

% Should be calculated once based on the original `vY`.
% In PCG each iteration `vY` changes, so it must be calculated as an
% auxiliary function.
% mIx = mIx ./ ((abs(mIx) .^ paramGamma) + paramEps);
% mIy = mIy ./ ((abs(mIy) .^ paramGamma) + paramEps);

mIx = mIx .* mAx;
mIy = mIy .* mAy;

mIx = conv2(mIx, [-1, 1], 'full');
mIy = conv2(mIy, [-1; 1], 'full');

vX = vY + (paramLambda * (mIx(:) + mIy(:)));


end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

