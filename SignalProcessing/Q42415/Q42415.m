% Signal Processing Q42415
% https://dsp.stackexchange.com/questions/42415
% How to calculate kernel in guided image filter?
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     18/07/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

vFilterRadius   = [1, 3, 5];
vRegFctr        = [0.05, 0.2, 0.5];

mI = imread('Lena256.png');
mI = double(mI) ./ 255;


%% Apply Local Linear Filter

hFigure         = figure('Position', [100, 100, 760, 720]);
hAxes           = axes();
hImageObject    = image(repmat(mI, [1, 1, 3]));
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
set(get(hAxes, 'Title'), 'String', {['Input Image - Lena']}, ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTick', [], 'YTick', []);


hFigure = figure('Position', figPosLarge);

for ii = 1:length(vFilterRadius)
    for jj = 1:length(vRegFctr)
        
        filterRadius    = vFilterRadius(ii);
        regFctr         = vRegFctr(jj);
        mO              = ApplyLocalLinearFilter(mI, filterRadius, regFctr);
        
        hAxes           = subplot_tight(length(vFilterRadius), length(vRegFctr), ((ii - 1) * length(vFilterRadius)) + jj, [0.05, 0.05]);
        hImageObject    = image(repmat(mO, [1, 1, 3]));
        set(hAxes, 'DataAspectRatio', [1, 1, 1]);
        set(hAxes, 'XTick', [], 'YTick', []);
        set(get(hAxes, 'Xlabel'), 'String', {['Radius - ', num2str(filterRadius), ', Reg Factor - ', num2str(regFctr)]}, ...
            'FontSize', fontSizeTitle);
    end
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

