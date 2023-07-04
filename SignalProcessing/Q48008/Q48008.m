% StackExchange Signal Processing Q48008
% https://dsp.stackexchange.com/questions/48008
% Find the Mid Point of a Worm Like Object
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     30/06/2022
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Simulation Constants




%% Simulation Parameters

% Array
imgUrl = 'https://i.stack.imgur.com/RzlYW.png';


%% Generate / Load Data

mI = imread(imgUrl);
mB = imbinarize(mI);
mB = mB(:, :, 2); %<! The channel which binarize well


%% Algorithm

% Create the Skeleton
% The assumption each pixel is connected to up to 2 pixels.
% Walking on the cahin means there is a single linked pixel which is not
% where we came from.
mS = bwskel(mB);

% Kernel to count the neighbors
mK = ones(3, 3);
mK(2, 2) = 0;

% Count the neighbors of each pixel in the skeleton
mN = conv2(mS, mK, 'same');
mN = mN .* mS;

% Algorithm:
% - Find one of the edges of the chain.
% - Walk over the neighbors without ever going back.

% Matrix of pixels visited
mC = zeros(size(mS), 'logical');

% Number of pixels in the chain
ll  = 0;
% The indices of element in the chain
vII = zeros(numel(mC), 1);
vJJ = zeros(numel(mC), 1);

idxFirst = find(mN == 1, 1, 'first');
[ii, jj] = ind2sub(size(mN), idxFirst);
ll  = ll + 1;
vII(ll) = ii;
vJJ(ll) = jj;
mC(ii, jj) = 1;

[mm, nn] = FindNext(mS, mC, ii, jj);
ll  = ll + 1;
vII(ll) = mm;
vJJ(ll) = nn;
mC(mm, nn) = 1;

while(mN(mm, nn) ~= 1) %<! End of chain
    ii = mm;
    jj = nn;
    [mm, nn] = FindNext(mS, mC, ii, jj);
    ll  = ll + 1;
    vII(ll) = mm;
    vJJ(ll) = nn;
    mC(mm, nn) = 1;
end

% Middle of the chain
middleIdx = round(ll / 2);
% Indices of the middle point
ii = vII(middleIdx);
jj = vJJ(middleIdx);


%% Display Results

figureIdx = figureIdx + 1;

hF = figure('Position', figPosLarge);
hA  = axes(hF);
set(hA, 'DataAspectRatio', [1, 1, 1]);
hImgObj = imagesc(mB);
set(get(hA, 'Title'), 'String', {['Input Binary Image']}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

figureIdx = figureIdx + 1;

hF = figure('Position', figPosLarge);
hA  = axes(hF);
set(hA, 'DataAspectRatio', [1, 1, 1]);
hImgObj = imagesc(mS);
set(get(hA, 'Title'), 'String', {['Skeleton Image']}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


figureIdx = figureIdx + 1;

hF = figure('Position', figPosLarge);
hA  = axes(hF);
set(hA, 'DataAspectRatio', [1, 1, 1]);
hImgObj  = imagesc(mB);
set(hA, 'NextPlot', 'add'); %<! Must be after displaying the image
hSctterObj = scatter(jj, ii, 'r', 'filled', 'DisplayName', 'Middle Point');
set(hSctterObj, 'SizeData', 20);
set(get(hA, 'Title'), 'String', {['Binary Image with Middle Point']}, ...
    'FontSize', fontSizeTitle);
hLegend = ClickableLegend();

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end




% set(hA, 'NextPlot', 'add');


%% Auxilizary Functions

function [ mm, nn ] = FindNext(mW, mC, ii, jj)

for pp = -1:1
    mm = ii + pp;
    for qq = -1:1
        nn = jj + qq;
        % if((pp == 0) && (qq == 0))
        %     % Don't say put
        %     continue;
        % end
        if(mW(mm, nn) && ~mC(mm, nn))
            return;
        end
    end
end


end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

