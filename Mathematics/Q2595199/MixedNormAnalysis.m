% Mathematics Q2595199
% https://math.stackexchange.com/questions/2595199
% Proximal Mapping of Least Squares with L1 and L2 Mixed Norm Regularization (Elastic Net)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     13/03/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

paramLam1 = 0.75;
paramLam2 = 1;

numElements = 2;
numSamples  = 15;


%% Generate Data

vGrid = linspace(-3, 3, numSamples);

vB = zeros([numElements, 1]);
mB = zeros([numElements, numSamples * numSamples]);
mX = zeros([numElements, numSamples * numSamples]);


hObjFun = @(vX) (0.5 * sum((vX - vB) .^ 2)) + (paramLam1 * norm(vX, 1)) + (paramLam2 * norm(vX, 2));
hL2SubGrad = @(vX) (norm(vX, 2) == 0) * (vX ./ (norm(vX, 2) + ((norm(vX, 2) == 0) * 1e-7)));


%% Solution by CVX

itrIdx = 0;

for ii = 1:numSamples
    vB(1) = vGrid(ii);
    for jj = 1:numSamples
        itrIdx = itrIdx + 1;
        
        vB(2) = vGrid(jj);
        
        cvx_begin('quiet')
        % cvx_precision('best');
        variable vX(numElements);
        minimize( (0.5 * square_pos(norm(vX - vB, 2))) + (paramLam1 * norm(vX, 1)) + (paramLam2 * norm(vX, 2)) );
        cvx_end
        
        mB(:, itrIdx) = vB;
        mX(:, itrIdx) = vX;
        
        disp(['Finished Sample #', num2str(itrIdx, figureCounterSpec), ' Ouf of ', num2str(numSamples * numSamples)]);
    end
    
    
end

vBB = mB(1, :) ./ mB(2, :);
vXX = mX(1, :) ./ mX(2, :);


%% Display Result

% hFigure     = figure('Position', figPosLarge);
% hAxes       = axes();
% hLineSeries = plot(vB, mX.');
% set(hLineSeries, 'LineWidth', lineWidthNormal);
% set(get(hAxes, 'Title'), 'String', ['Least Squares with Mixed Norm Regularization'], ...
%     'FontSize', fontSizeTitle);
% set(get(hAxes, 'XLabel'), 'String', 'b', ...
%     'FontSize', fontSizeAxis);
% set(get(hAxes, 'YLabel'), 'String', 'x', ...
%     'FontSize', fontSizeAxis);
% set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

