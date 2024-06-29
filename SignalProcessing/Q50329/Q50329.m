% StackExchange Signal Processing Q50329
% https://dsp.stackexchange.com/questions/50329/
% Automatic Image Enhancement of Images of Scanned Documents
% References:
%   1.  aa
% Remarks:
%   1.  Image - Certificate of Arrival for Berta Werner (Wikipdia).
%       Taken from https://commons.wikimedia.org/wiki/File:Certificate_of_Arrival_for_Berta_Werner._-_NARA_-_282038.jpg.
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     29/06/2018  Royi
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

imageFileName = 'InputImage.jpg'; %<! Source - WikiMedia (Certificate of Arrival for Berta Werner).

vParamLambda = linspace(0, 5, 8);


%% Load & Generate Data

mI = im2double(imread(imageFileName));
mI = mI(11:410, 201:600, 1);

% mI = mI(11:20, 201:210, 1);

numRows     = size(mI, 1);
numCols     = size(mI, 2);
numPixels   = numRows * numCols;

mO = zeros([numRows, numCols, length(vParamLambda)]); %<! Output

vDx = [1, -1]; %<! Matrix is doing Correlation, this for Convolution
vDy = [1; -1];

% Sanity Check
mIxRef = conv2(mI, vDx, 'valid');
mIyRef = conv2(mI, vDy, 'valid');

% mIx = reshape(mDx * mI(:), numRows, numCols - 1);
% mIy = reshape(mDy * mI(:), numRows - 1, numCols);

mDh = sparse(numPixels - numCols, numPixels);
mDv = sparse(numPixels - numRows, numPixels);

% mDx = zeros(numPixels - numCols, numPixels);
% mDy = zeros(numPixels - numRows, numPixels);

tic();
colShift = 0;
for ii = 1:(numPixels - numRows)
    if(mod(ii + colShift, numRows) == 0)
        % Take care of cases where the pixel is the last pixel in the
        % column (Resided on the last row of the image hence no valid
        % vertical derivative).
        colShift = colShift + 1;
    end
    mDv(ii, ii + colShift) = -1;
    mDv(ii, ii + colShift + 1) = 1;
end

for ii = 1:(numPixels - numCols)
    mDh(ii, ii) = -1;
    mDh(ii, ii + numCols) = 1;
end
toc();


mIx = reshape(mDh * mI(:), numRows, numCols - 1);
mIy = reshape(mDv * mI(:), numRows - 1, numCols);

mE = mIy - mIyRef;
disp(['Maximum Absolute Error Between Matrix Form to Convolution (Vertical Derivative) - ', num2str(max(abs(mE(:))))]);

mE = mIx - mIxRef;
disp(['Maximum Absolute Error Between Matrix Form to Convolution (Horizontal Derivative) - ', num2str(max(abs(mE(:))))]);

mDDh    = mDh.' * mDh;
mDDv    = mDv.' * mDv;
mDD     = mDDh + mDDv; %<! Laplacian

% The following are equivalent:
% mO = reshape(mDD * mI(:), numRows, numCols);
% mO = conv2(conv2(mI, vDx, 'valid'), vDx(end:-1:1), 'full') + conv2(conv2(mI, vDy, 'valid'), vDy(end:-1:1), 'full');
% Pay attention that in real Laplacian we would apply convolution twice
% while in the above we apply its Adjoint (Correlation) hence the filter is
% practically [-1, 2, -1] while for Laplacian it is [1, -2, 1].
% In parts:
% mO = reshape(mDDh * mI(:), numRows, numCols);
% mO = conv2(conv2(mI, vDx, 'valid'), vDx(end:-1:1), 'full');
% mO = reshape(mDDv * mI(:), numRows, numCols);
% mO = conv2(conv2(mI, vDy, 'valid'), vDy(end:-1:1), 'full');

% Verification Code:
% mII         = speye(numPixels);
% paramLambda = 0.5;
% mXRef       = reshape((mII + (paramLambda * mDD)) * mI(:), numRows, numCols);
% mX          = mI + (paramLambda * CalcImageLaplacian(mI));
% mE          = mXRef - mX;
% max(abs(mE(:)))


%% Analysis

mII = speye(numPixels);
vB  = ones([numPixels, 1]);

tic();
for ii = 1:length(vParamLambda)
    paramLambda = vParamLambda(ii);
    mO(:, :, ii) = reshape((mII + (paramLambda * mDD)) \ ((paramLambda * mDD * mI(:)) + vB), numRows, numCols);
end
toc();


%% Display Results

figureIdx = figureIdx + 1;

hFigure         = figure('Position', [100, 100, 1100, 1100]);

hAxes     = subplot(sqrt(length(vParamLambda) + 1), sqrt(length(vParamLambda) + 1), 1);
hImageObj = image(repmat(mI, [1, 1, 3]));
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
set(get(hAxes, 'Title'), 'String', {['Input Image']}, ...
    'FontSize', fontSizeTitle);

for ii = 1:length(vParamLambda)
    hAxes     = subplot(sqrt(length(vParamLambda) + 1), sqrt(length(vParamLambda) + 1), ii + 1);
    hImageObj = image(repmat(mO(:, :, ii), [1, 1, 3]));
    set(hAxes, 'DataAspectRatio', [1, 1, 1]);
    set(get(hAxes, 'Title'), 'String', {['Output Image - \lambda - ', num2str(vParamLambda(ii))]}, ...
        'FontSize', fontSizeTitle);
end

if(generateFigures == ON)
    saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

