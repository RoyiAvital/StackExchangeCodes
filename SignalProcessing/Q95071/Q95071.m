% StackExchange Signal Processing Q95071
% https://dsp.stackexchange.com/questions/95071
% Build the Laplacian Matrix of Edge Preserving Multiscale Image Decomposition based on Local Extrema
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     16/09/2024
%   *   First release.


%% Settings

subStreamNumberDefault  = 79;
keepFigures             = false;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Parameters

% Image from the paper (Cropped and resized)
imgUrl = 'https://i.postimg.cc/85Jjs9wJ/Flowers.png'; %<! https://i.imgur.com/PckT6jF.png
% imgUrl = 'https://raw.githubusercontent.com/yafangshih/EdgePreserving-Blur/refs/heads/master/data/input/taipei101.jpg';

paramK = 3; %<! Radius of the local extrema
paramN = 1; %<! Radius of the local Laplacian

forceSymmetricGraph = FALSE; %<! Symmetry ruins edge preservation

epsVal = 1e-5;

% Validation Function: ii, jj ref pixel location, mm, nn neighbor pixel shift
hV = @(ii, jj, mm, nn) (abs(mm) <= 1) && (abs(nn) <= 1) && ((mm ~= 0) || (nn ~= 0)); %<! 8 Point connectivity, excludes center for no cyclic graph
% hV = @(ii, jj, mm, nn) (mm * nn == 0) && ((mm ~= 0) || (nn ~= 0)); %<! 4 Point connectivity, excludes center for no cyclic graph

% Weighing Function
% MATLAB removes zero values automatically on each value insertion.
% Usind added value to prevent such case.
hW = @(valI, valN) abs(valI - valN) + epsVal; %<! Reference value vs. neighbor (r <-> s in the paper)


%% Generate / Load Data

mI = im2double(imread(imgUrl));
mI = mean(mI, 3); %<! RGB Image

numRows = size(mI, 1);
numCols = size(mI, 2);
numPx = numRows * numCols;


%% Analysis
hRunTime = tic();

vPxInd = 1:numPx;

localLen = (2 * paramK) + 1; %<! k in the paper

% Local Extrema
mLocalKValue = ordfilt2(mI, localLen * localLen - localLen + 1, ones(localLen), 'symmetric');
mLocalMax = mI >= mLocalKValue; %<! Local Maximum
mLocalKValue = ordfilt2(mI, localLen, ones(localLen), 'symmetric');
mLocalMin = mI <= mLocalKValue; %<! Local Minimum

% Graph
% Local variance image
mV = stdfilt(mI, true(2 * paramN + 1));
mV = mV .* mV;

mGV = CalcMGV(mI);

mW = BuildGraphMatrix(mI, hV, hW, paramN);

% Exponential Weights
[vR, vC, vVals] = find(mW);
for ii = 1:length(vR)
    localVar  = 0.6 * mV(vR(ii)); %<! The row is the reference pixel index
    mgVal     = mGV(vR(ii));
    localVar  = max(localVar, -mgVal / log(0.01));
    localVar  = max(localVar, 0.000002) / 2;
    vVals(ii) = exp(-(vVals(ii) * vVals(ii)) / (2 * localVar)); %<! Exponent function
end
mW = sparse(vR, vC, vVals, numPx, numPx);

if(forceSymmetricGraph)
    mW = mW + mW.';
end

mW = NormalizeRows(mW);


% % Graph Laplacian
mD = diag(sum(mW, 2)); %<! Degree Matrix
mL = mD - mW; %<! Laplacian Matrix

% Interpolate Maximum
vV = find(mLocalMax);
vU = setdiff(vPxInd(:), vV);

mLu = mL(vU, vU); %<! The Laplacian sub matrix to optimize by
mR  = mL(vU, vV);
oLu = decomposition(mLu);

vXv  = mI(vV); %<! Anchor values
vXu = -(oLu \ (mR * vXv));

vX = zeros(numPx, 1);
vX(vU) = vXu;
vX(vV) = vXv;
mXMax = reshape(vX, numRows, numCols);

% Interpolate Minimum
vV = find(mLocalMin);
vU = setdiff(vPxInd(:), vV);

mLu = mL(vU, vU); %<! The Laplacian sub matrix to optimize by
mR  = mL(vU, vV);
oLu = decomposition(mLu);

vXv  = mI(vV); %<! Anchor values
vXu = -(oLu \ (mR * vXv));

vX = zeros(numPx, 1);
vX(vU) = vXu;
vX(vV) = vXv;
mXMin = reshape(vX, numRows, numCols);


mX = 0.5 * (mXMax + mXMin);

runTime = toc(hRunTime);
disp(['Run Time: ', num2str(runTime, '%0.2f'), ' [Sec]']);


%% Display Results

figureIdx = figureIdx + 1;

[hF, ~, ~] = PlotImages(permute(cat(3, mI, mXMax, mXMin, mX), [3, 1, 2]), ...
    'cPlotTitle', {'Input Image', 'Local Maxima', 'Local Minima', 'Output (Mean)'}, ...
    'supTitle', {['Local Extrema Edge Preserving Blur, k = ', num2str(paramK)]}, ...
    'vSize', [2, 2]);

if(generateFigures == ON)
    % saveas(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end



%% Auxiliary Functions

function [ mW ] = NormalizeRows( mW )

numRows = size(mW, 1);
numCols = size(mW, 2);
numNnz  = nnz(mW);

[vR, vC, vV] = find(mW);
vRowSum = zeros(size(mW, 1), 1);
for ii = 1:numNnz
    vRowSum(vR(ii)) = vRowSum(vR(ii)) + vV(ii);
end

for ii = 1:numRows
    vRowSum(ii) = ((vRowSum(ii) == 0) * 0) + ((vRowSum(ii) ~= 0) * vRowSum(ii));
end

for ii = 1:numNnz
    vV(ii) = vV(ii) / vRowSum(vR(ii));
end

mW = sparse(vR, vC, vV, numRows, numCols);


end


function [ mO ] = CalcMGV( mI )

mP = padarray(mI, [1, 1], 'both', 'replicate');
mC = im2col(mP, [3, 3], 'sliding');
mC = mC - mC(5, :);
mC(5, :) = 1000;
mC = mC .* mC;
mO = col2im(min(mC), [3, 3], size(mP), 'sliding');


end






%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

