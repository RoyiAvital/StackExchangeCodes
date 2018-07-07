% StackExchange Signal Processing Q50003
% https://dsp.stackexchange.com/questions/50003/
% Automatic Image Enhancement of Images of Scanned Documents
% References:
%   1.  aa
% Remarks:
%   1.  Image - Certificate of Arrival for Berta Werner (Wikipdia).
%       Taken from https://commons.wikimedia.org/wiki/File:Certificate_of_Arrival_for_Berta_Werner._-_NARA_-_282038.jpg.
%   2.  This solves teh same problem using Conjugate Gradient.
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


%% Analysis

mII = speye(numPixels);
vB  = ones([numPixels, 1]);

tic();
for ii = 1:length(vParamLambda)
    paramLambda = vParamLambda(ii);
    
    hAFun   = @(vX) vX + (paramLambda * reshape(CalcImageLaplacian(  reshape(vX, numRows, numCols) ), numPixels, 1));
    vB      = paramLambda * reshape(CalcImageLaplacian(mI), [], 1) + ones([numPixels, 1]);
    
    mO(:, :, ii) = reshape(pcg(hAFun, vB), numRows, numCols);
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

