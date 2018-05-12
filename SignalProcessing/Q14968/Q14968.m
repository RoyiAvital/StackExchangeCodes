% StackExchange Signal Processing Q14968
% https://dsp.stackexchange.com/questions/14968/
% Using Total Variation Denoising to Clean Accelerometer Data
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     12/05/2018  Royi
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numSamples      = 300;
numAtoms        = 100; %<! Number of Atoms in Dictionary
numAtomsSignal  = 10; %<! Number of Atoms used to generate signal

noiseStd    = 0.1;
paramLambda = 0.25;

numIterations   = 5000;
stepSize        = 0.001;


%% Generate Data

mA = zeros([numSamples, numAtoms]);

% Building dictionary of steps of different length
for ii = 1:numAtoms
    firstIdx    = randi([1, numSamples - 1], [1, 1]);
    lastIdx     = randi([firstIdx + 1, numSamples], [1, 1]);
    
    mA(firstIdx:lastIdx, ii) = 1;
end

vAtomIdx        = randperm(numAtoms, numAtomsSignal);
vX              = zeros([numAtoms, 1]);
vX(vAtomIdx)    = sign(rand([numAtomsSignal, 1]) - 0.5);

vY = mA * vX;
vB = vY + (noiseStd * randn([numSamples, 1]));

% Difference Matrix
mD = zeros([(numSamples - 1), numSamples]);

for ii = 1:(numSamples - 1)
    mD(ii, ii)      = -1;
    mD(ii, ii + 1)  = 1;
end


%% Solutions for the Total Variation Problem (CVX, Sub Gradient)

% Reference by CVX
cvx_begin('quiet')
    cvx_precision('best');
    variable vZ(numSamples)
    minimize( (0.5 * square_pos(norm(vZ - vB, 2))) + (paramLambda * norm(mD * vZ, 1)) );
cvx_end

% Solution by Sub Gradient

vZSgm = zeros([numSamples, 1]);

for ii = 1:numIterations
    vG      = (vZSgm - vB) + (paramLambda * (mD.' * sign(mD * vZSgm)));
    vZSgm   = vZSgm - (stepSize * vG);
end

mI      = eye(numSamples);
vZTik   = (mI + (2 * paramLambda * (mD.' * mD))) \ vB;


%% Display Results

figureIdx = figureIdx + 1;

hFigure         = figure('Position', figPosLarge);
hAxes           = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries  = line(1:numSamples, [vY, vB, vZ, vZSgm, vZTik]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2), 'LineWidth', lineWidthThin);
set(hLineSeries(5), 'LineWidth', lineWidthThin);
set(hLineSeries(5), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Edge Preserving Denoising']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Samples Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['Noisy signal'], ['Total Variation - CVX'], ['Total Variation - SGM'], ['Tikhonov Regularization']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


figureIdx = figureIdx + 1;

hFigure         = figure('Position', figPosLarge);
hAxes           = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries  = line(1:numSamples, [vY, vZSgm, vZTik]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(3), 'LineWidth', lineWidthThin);
set(get(hAxes, 'Title'), 'String', {['Edge Preserving Denoising']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Samples Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['Total Variation - SGM'], ['Tikhonov Regularization']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

