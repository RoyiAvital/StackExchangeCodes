% StackExchange Signal Processing Q18673
% https://dsp.stackexchange.com/questions/18673
% Image Compression Using the SVD.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     14/02/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants




%% Parameters

imageFileName   = 'Lena512Color.png';
vEnergyThr      = [1, 0.95, 0.85, 0.75, 0.65, 0.55];
blockRadius     = 3;


%% Generate / Load Data

mI = im2double(imread(imageFileName));


%% Analysis

numImg = length(vEnergyThr);

figureIdx   = figureIdx + 1;

figWidth = (numImg * 256) + ((numImg + 1) * 40);

hF = figure('Position', [100, 100, figWidth, 400], 'Units', 'pixels');

for ii = 1:numImg
    
    energyThr   = vEnergyThr(ii);
    mO          = CompressImageSvd(mI, energyThr, blockRadius);
    
    leftMargin   = 40 + ((ii - 1) * (256 + 40));
    hA           = axes(hF, 'Position', [leftMargin, 40, 256, 256]);
    hImageObject = image(mO);
    set(get(hA, 'XLabel'), 'String', {['Energy Threhold = ', num2str(energyThr)]}, ...
        'FontSize', fontSizeAxis);
    set(hA, 'XTick', []);
    set(hA, 'XTickLabel', []);
    set(hA, 'YTick', []);
    set(hA, 'YTickLabel', []);
    set(hA, 'DataAspectRatio', [1, 1, 1]);
    set(hA, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
    drawnow();
end

if(generateFigures == ON)
    % saveas(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Display Results


%% Auxiliary Functions




%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

