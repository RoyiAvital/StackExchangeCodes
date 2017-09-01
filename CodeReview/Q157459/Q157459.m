% Mathematics Q157459
% https://codereview.stackexchange.com/questions/157459
% Image Compression Using the Singular Value Decomposition (SVD) with MATLAB
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     01/09/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Data Parameters

imageFileName   = 'Lena512Color.png';
vEnergyThr      = [1, 0.95, 0.85, 0.75, 0.65, 0.55];
blockRadius     = 3;


%% Load Data

mI = im2double(imread(imageFileName));


%% Analysis


figureIdx   = figureIdx + 1;
hFigure     = figure('Position', [100, 100, 1400, 900]);

for ii = 1:length(vEnergyThr)
    
    energyThr   = vEnergyThr(ii);
    mO          = CompressImageSvd(mI, energyThr, blockRadius);
    
    hAxes           = subplot_tight(2, 3, ii, [0.06, 0.06]);
    % hAxes           = subplot(1, 3, ii);
    hImageObject    = image(mO);
    set(get(hAxes, 'XLabel'), 'String', {['Energy Threhold = ', num2str(energyThr)]}, ...
        'FontSize', fontSizeAxis);
    set(hAxes, 'XTick', []);
    set(hAxes, 'XTickLabel', []);
    set(hAxes, 'YTick', []);
    set(hAxes, 'YTickLabel', []);
    set(hAxes, 'DataAspectRatio', [1, 1, 1]);
    % set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
    drawnow();
end

saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png'], 'png');


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

