% StackExchange Signal Processing Q51386
% https://dsp.stackexchange.com/questions/51386
% Extended Kalman Filter (EKF) for Non Linear (Coordinate Conversion - Polar to Cartesian) Measurements and Linear Predictions
% References:
%   1.  Extended Kalman Filter (Wikipedia) - https://en.wikipedia.org/wiki/Extended_Kalman_filter.
%   1.  Bayesian Filtering for Dynamic Systems with Applications to Tracking (Page 35).
%   2.  Estimation with Applications to Tracking and Navigation (Page 397).
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     21/08/2018
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;

EKF_JACOBIAN_METHOD_ANALYTIC    = 1;
EKF_JACOBIAN_METHOD_NUMERIC     = 2;
UKF                             = 3;


%% Simulation Parameters

numMeasurements = 100;

dT = 1.5;

vX0 = [1500; -7.5; 1500; -5]; %!< x, vx, y, vy
mF  = [1, dT, 0, 0; 0, 1, 0, 0; 0, 0, 1, dT; 0, 0, 0, 1];
hF = @(vX) mF * vX;
hH  = @(vX) [norm(vX([1, 3])); atan(vX(3) / vX(1))]; %<! Usage of 'norm()' won't work with Complex Step Derivative
hH  = @(vX) [sqrt(sum(vX([1, 3]) .^ 2)); atan(vX(3) / vX(1))]; %<! Usage of 'norm()' won't work with Complex Step Derivative
hY = @(vZ) [vZ(1) * cos(vZ(2)); vZ(1) * sin(vZ(2))];

mP0 = diag([1, 0.1, 1, 0.1]);

mQ = [0.35, 0, 0, 0; 0, 0.05, 0, 0; 0, 0, 0.35, 0; 0, 0, 0, 0.05];
mR = [0.1, 0; 0, (0.3 / 180) * pi];

kalmanMethod = EKF_JACOBIAN_METHOD_ANALYTIC;
% kalmanMethod = EKF_JACOBIAN_METHOD_NUMERIC;
kalmanMethod = UKF;


%% Generate Data

modelDim    = size(vX0, 1);
measDim     = size(mR, 1);

mQChol = chol(mQ, 'lower');
mRChol = chol(mR, 'lower');

mX = zeros(modelDim, numMeasurements + 1);
mX(:, 1) = vX0 + (chol(mP0, 'lower') * randn(modelDim, 1));
vZ0 = hH(vX0) + (mRChol * randn(measDim, 1));
mZ = zeros(measDim, numMeasurements + 1);
mZ(:, 1) = vZ0;
mY = zeros(measDim, numMeasurements + 1);
mY(:, 1) = hY(mZ(:, 1));

for ii = 1:numMeasurements
    mX(:, ii + 1) = (mF * mX(:, ii)) + (mQChol * randn(modelDim, 1));
    mZ(:, ii + 1) = hH(mX(:, ii + 1)) + (mRChol * randn(measDim, 1));
    mY(:, ii + 1) = hY(mZ(:, ii + 1));
end


%% Extended Kalman Filter Estimation

vXEst = vX0 + (chol(mP0, 'lower') * randn(modelDim, 1));
mXEst = zeros(modelDim, numMeasurements + 1);
mXEst(:, 1) = vXEst;
tP = zeros(modelDim, modelDim, numMeasurements + 1);
tP(:, :, 1) = mP0;

mH = zeros(measDim, modelDim);

for ii = 1:numMeasurements
    if(kalmanMethod == EKF_JACOBIAN_METHOD_ANALYTIC)
        % Analytic Jacobian
        vX = hF(mXEst(:, ii));
        mH(:) = [vX(1) / norm(vX([1, 3])); -vX(3) / (norm(vX([1, 3])) ^ 2); zeros(2, 1); vX(3) / norm(vX([1, 3])); vX(1) / (norm(vX([1, 3])) ^ 2); zeros(2, 1)];
        
        % Numerical Validation
        % mE = mH - CalcFunJacob(vX, hH, 4, 1e-7);
        % max(abs(mE(:)))
        
        % [mXEst(:, ii + 1), tP(:, :, ii + 1)] = ApplyKalmanFilterIteration(mXEst(:, ii), tP(:, :, ii), mZ(:, ii + 1), hF, hH, mQ, mR, mF, mH);
        [mXEst(:, ii + 1), tP(:, :, ii + 1)] = ApplyUnscentedKalmanFilterIteration(mXEst(:, ii), tP(:, :, ii), mZ(:, ii + 1), hF, hH, mQ, mR);
    end
    
    if(kalmanMethod == EKF_JACOBIAN_METHOD_NUMERIC)
        % Numeric Jacobian
        [mXEst(:, ii + 1), tP(:, :, ii + 1)] = ApplyKalmanFilterIteration(mXEst(:, ii), tP(:, :, ii), mZ(:, ii + 1), hF, hH, mQ, mR);
    end
    
    if(kalmanMethod == UKF)
        [mXEst(:, ii + 1), tP(:, :, ii + 1)] = ApplyUnscentedKalmanFilterIteration(mXEst(:, ii), tP(:, :, ii), mZ(:, ii + 1), hF, hH, mQ, mR);
    end
end


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries = plot(mX(1, :), mX(3, :));
set(hLineSeries, 'LineWidth', lineWidthThin, 'Marker', '*', 'Color', mColorOrder(1, :));
hLineSeries = plot(mY(1, :), mY(2, :));
set(hLineSeries, 'LineStyle', 'none', 'Marker', '*', 'Color', mColorOrder(2, :));
hLineSeries = plot(mXEst(1, :), mXEst(3, :));
set(hLineSeries, 'LineStyle', 'none', 'Marker', '*', 'Color', mColorOrder(3, :));
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
set(hAxes, 'Xlim', [0, 2000], 'YLim', [0, 2000]);
set(get(hAxes, 'Title'), 'String', {['Extended Kalman Estimation - Cartesian Coordinate Model and Polar Coordinate Measurement']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['x [Meters]']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['y [Meters]']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Ground Truth'], ['Measurement'], ['Estimation']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

