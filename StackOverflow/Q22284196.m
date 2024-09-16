% Stack Overflow Q22284196
% https://stackoverflow.com/questions/22284196
% Remove Noise as a Pre Processing of Edge Detection with Edge Preserving Filter
% References:
%   1.  
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     16/09/2024
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Parameters

% The post image link: https://i.sstatic.net/8mSdV.jpg
imgUrl = 'https://i.imgur.com/5Fag3vS.png'; %<! https://i.postimg.cc/VN4DNKzM/image.png

vFilterRadius = 1:6;


%% Load Data

mI = im2double(imread(imgUrl));
mI = mI(:, :, 1);

numRows = size(mI, 1);
numCols = size(mI, 2);


%% Analysis

vFilterLen = 2 * vFilterRadius + 1;
numFilters = length(vFilterLen);

tO = zeros(numRows, numCols, numFilters);

for ii = 1:numFilters
    tO(:, :, ii) = medfilt2(mI, [vFilterLen(ii), vFilterLen(ii)], 'symmetric');
end


%% Display Data

cPlotTile = cell(1, numFilters);
for ii = 1:numFilters
    cPlotTile{ii} = ['Radius: ', num2str(vFilterRadius(ii))];
end

[hF, ~, ~] = PlotImages(permute(tO, [3, 1, 2]), 'cPlotTitle', cPlotTile, 'vSize', [2, 3]);
sgtitle(hF, 'Median Filter');



%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

