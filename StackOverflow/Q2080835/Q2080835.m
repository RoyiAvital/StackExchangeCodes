% StackOverflow Q2080835
% https://stackoverflow.com/questions/2080835
% Deriving the Inverse Filter of Image Convolution Kernel
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     14/01/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 0;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;
generateImages  = OFF;

OPERATION_MODE_CONVOLUTION = 1;
OPERATION_MODE_CORRELATION = 2;

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Simulation Parameters

operationMode = OPERATION_MODE_CONVOLUTION;
convShape = CONVOLUTION_SHAPE_VALID;

% The Input Kernel - F
numRowsF = 11;
numColsF = 7;

% The Inverse Kernel - G
numRowsG = 201;
numColsG = 201;

numIteraions    = 50000;
stepSize        = 5e-5;


%% Generate Data

numRowsH = numRowsF + numRowsG - 1;
numColsH = numColsF + numColsG - 1;

mF = rand(numRowsF, numColsF);
mG = ones(numRowsG, numColsG); %<! Initial condition


%% Verify Gradient
% Using Random h Filter

mH = rand(numRowsH, numColsH);

hObjFun = @(mG) 0.5 * sum((conv2(mF, mG, 'full') - mH) .^ 2, 'all');

mObjFunGrad = conv2(conv2(mF, mG, 'full') - mH, mF(end:-1:1, end:-1:1), 'valid');

mObjFunGradNum = zeros(size(mG));
mTmp = zeros(size(mObjFunGradNum));

derEps = 5e-6;

% Numeric Gradient
for ii = 1:numel(mObjFunGradNum)
    mTmp(ii)            = derEps;
    mObjFunGradNum(ii)  = (hObjFun(mG + mTmp) - hObjFun(mG)) / derEps;
    mTmp(ii)            = 0;
end

mE = mObjFunGradNum - mObjFunGrad;
gradError = max(abs(mE(:)));


%% Derive the Inverse

% The Target Kernel - Discrete Delta
mH = zeros(numRowsH, numColsH);
mH(ceil(numRowsH / 2), ceil(numColsH / 2)) = 1; %<! Delta

% Gradient Descent (Could improved with Accelerated Gradient Descent)
for ii = 1:numIteraions
    mObjFunGrad = conv2(conv2(mF, mG, 'full') - mH, mF(end:-1:1, end:-1:1), 'valid');
    mG          = mG - (stepSize * mObjFunGrad);
end

mA = conv2(mG, mF, 'full');
mE = abs(mA - mH);
inverseError = max(mE(:));


%% Analysis

disp(['Analytic Gradient vs. Numeric Gradient - Maximum Absolute Deviation - ', num2str(gradError)]);
disp(['Inverse Filter - Maximum Deviation - ', num2str(inverseError)]);


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes();
hImgObj = imshow(mA, []);
set(get(hAxes, 'Title'), 'String', {['Convolution Result']}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

